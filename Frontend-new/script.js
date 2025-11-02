const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');

const userMessageTemplate = document.getElementById('user-message-template');
const botMessageTemplate = document.getElementById('bot-message-template');
const typingIndicatorTemplate = document.getElementById('typing-indicator-template');

// Load from sessionStorage on page load
let currentCollectionName = sessionStorage.getItem('activeCollectionName') || null;
let currentFileName = sessionStorage.getItem('activeFileName') || null;

const chatHistory = [{
    role: "model",
    parts: [{ text: "You are a helpful and friendly AI assistant. Upload a PDF to start asking questions about it." }]
}];

// This code runs once the chat page DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (currentCollectionName) {
        // If we loaded a doc, display the welcome message
        const welcomeMessage = `**File Ready!** You are now chatting with \`${currentFileName}\`. Ask me anything about it.`;
        displayMessage(botMessageTemplate, welcomeMessage);
        
        // Also update chat history
        chatHistory.push({ role: "user", parts: [{ text: `I have just uploaded ${currentFileName}.` }] });
        chatHistory.push({ role: "model", parts: [{ text: `Great! I'm ready to answer questions about ${currentFileName}.` }] });
    
    }
});

// --- Voice Recording State and Objects ---
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Function to handle the upload of the recorded audio
const uploadAudioForTranscription = async (audioBlob) => {
    const uploadStatus = displayMessage(botMessageTemplate, `**Transcribing Audio...**`);
    const formData = new FormData();
    formData.append("audio_file", audioBlob, "user_recording.webm");

    try {
        const response = await fetch('/transcribe-audio/', {
            method: 'POST',
            body: formData
        });
        chatContainer.removeChild(uploadStatus);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Transcription server error.');
        }

        const data = await response.json();
        const transcribedText = data.text_english;

        if (!transcribedText || transcribedText.includes("Could not understand audio")) {
            displayMessage(botMessageTemplate, `**⚠️ ${transcribedText}**`);
            return;
        }
        messageInput.value = transcribedText; 
        sendMessage(); 

    } catch (error) {
        chatContainer.removeChild(uploadStatus);
        displayMessage(botMessageTemplate, `**⚠️ Transcription Error:** ${error.message}`);
        console.error('Transcription Upload Error:', error);
    }
};

// --- Function to call your RAG backend (gpt-oss) ---
const callRAGBackend = async (prompt) => {
    try {
        const response = await fetch('/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: prompt,
                collection_name: currentCollectionName 
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error ${response.status}`);
        }

        const data = await response.json();
        const text = data.answer;

        if (text) {
            chatHistory.push({ role: "user", parts: [{ text: prompt }] });
            chatHistory.push({ role: "model", parts: [{ text: text }] });
            return text;
        } else {
            // This will now be displayed, thanks to our other fix
            return "Sorry, I received an empty response from the RAG backend.";
        }

    } catch (error) {
        console.error("Error calling RAG backend:", error);
        return `Sorry, there was an error connecting to the document AI: ${error.message}`;
    }
};

// --- Utility function to display messages ---
const displayMessage = (template, text) => {
    const messageNode = template.cloneNode(true);
    messageNode.removeAttribute('id');
    messageNode.classList.remove('hidden');

    if (text) {
        const textElement = messageNode.querySelector('p');

        // --- THIS IS THE FIX ---
        // Check if this is a bot message template
        if (template.id === 'bot-message-template') {
            // If it's a bot, parse the text as Markdown
            // marked.parse() safely converts Markdown to HTML
            textElement.innerHTML = marked.parse(text);
        } else {
            // Otherwise (like a user message), insert it as plain text
            textElement.textContent = text;
        }
        // --- END OF FIX ---
    }

    chatContainer.appendChild(messageNode);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageNode;
}
// --- sendMessage (This now works correctly) ---
const sendMessage = async () => {
    const messageText = messageInput.value.trim();
    if (messageText === '') return;

    displayMessage(userMessageTemplate, messageText);
    messageInput.value = '';

    const typingIndicator = displayMessage(typingIndicatorTemplate);
    
    let botResponseText;

    if (currentCollectionName) {
        // If we have a PDF loaded, use the RAG backend
        console.log(`Sending to RAG backend with collection: ${currentCollectionName}`);
        botResponseText = await callRAGBackend(messageText);
    } else {
        // If no PDF is loaded, prompt the user.
        botResponseText = "Please upload a PDF document first to ask questions about it.";
    }

    chatContainer.removeChild(typingIndicator);
    
    // --- THIS IS THE FIX ---
    displayMessage(botMessageTemplate, botResponseText);
    // --- END THE FIX ---
};

// --- Voice Button Event Listener ---
voiceBtn.addEventListener('click', async () => {
    if (isRecording) {
        voiceBtn.style.color = '';
        isRecording = false;
        if (mediaRecorder) {
            mediaRecorder.stop();
        }
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' });
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                stream.getTracks().forEach(track => track.stop());
                uploadAudioForTranscription(audioBlob); 
            };

            mediaRecorder.start();
            voiceBtn.style.color = 'red';
            isRecording = true;
            //for debugging
            //displayMessage(botMessageTemplate, "**Recording...** Click again to stop. Please speak clearly in English.");
        } catch (error) {
            console.error('Microphone access denied or error:', error);
            displayMessage(botMessageTemplate, `**Error:** Could not access the microphone. ${error.message}`);
        }
    }
});

// --- Final Event Listeners ---
sendBtn.addEventListener('click', sendMessage);

messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});