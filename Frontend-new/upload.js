document.addEventListener('DOMContentLoaded', () => {
    const pdfInput = document.getElementById('pdf-input');
    const uploadBtn = document.getElementById('upload-btn');
    const statusMessage = document.getElementById('status-message');
    const fileNameDisplay = document.getElementById('file-name-display'); 

    if (!pdfInput || !uploadBtn || !statusMessage || !fileNameDisplay) {
        console.error("Upload script error: One or more required elements are missing.");
        return;
    }

    pdfInput.addEventListener('change', () => {
        if (pdfInput.files.length > 0) {
            const fileName = pdfInput.files[0].name;
            fileNameDisplay.textContent = fileName;
            statusMessage.textContent = ''; 
        } else {
            fileNameDisplay.textContent = 'None';
        }
    });

    uploadBtn.addEventListener('click', async () => {
        const file = pdfInput.files[0]; 

        if (!file) {
            statusMessage.textContent = '⚠️ Please select a file first.';
            return;
        }

        if (!file.name.toLowerCase().endsWith(".pdf")) {
            statusMessage.textContent = '⚠️ Invalid File: Please select a PDF file.';
            return;
        }

        uploadBtn.disabled = true;
        statusMessage.textContent = `Uploading ${file.name}...`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch('/upload-pdf/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                const errorMessage = errorData.detail || 'Upload failed due to a server error.';
                statusMessage.textContent = `❌ Upload Failed: ${errorMessage}`;
                uploadBtn.disabled = false; 
                return;
            }
            
            const data = await response.json();
            const collectionName = data.collection_name;

            if (!collectionName) {
                statusMessage.textContent = `❌ Upload OK, but server didn't return a collection name.`;
                uploadBtn.disabled = false;
                return;
            }

            // --- MODIFICATION: Use sessionStorage ---
            // This will be remembered as long as the tab is open
            sessionStorage.setItem('activeCollectionName', collectionName);
            sessionStorage.setItem('activeFileName', file.name);
            // --- END MODIFICATION ---

            statusMessage.textContent = '✅ Upload successful! Redirecting to chat...';
            
            setTimeout(() => {
                window.location.href = '/chat'; // Redirect to the chat page
            }, 1500);

        } catch (error) {
            statusMessage.textContent = '⚠️ Network Error: Could not connect to the server.';
            console.error('Upload Error:', error);
            uploadBtn.disabled = false; 
        }
    });
});