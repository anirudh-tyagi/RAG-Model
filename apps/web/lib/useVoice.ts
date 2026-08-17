"use client";

import { api } from "@/lib/api";
import { useCallback, useEffect, useRef, useState } from "react";

export interface UseVoice {
  recording: boolean;
  transcribing: boolean;
  error: string | null;
  supported: boolean;
  toggle: () => Promise<void>;
}

/**
 * Microphone capture → `/api/transcribe`.
 *
 * The recording is sent as-is (webm/opus) because faster-whisper decodes it
 * server-side; the old front end's blob went through pydub and an ffmpeg hop.
 */
export function useVoice(onTranscript: (text: string) => void): UseVoice {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [supported, setSupported] = useState(true);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  useEffect(() => {
    setSupported(
      typeof window !== "undefined" &&
        typeof window.MediaRecorder !== "undefined" &&
        !!navigator.mediaDevices?.getUserMedia,
    );
  }, []);

  const stop = useCallback(() => {
    recorder.current?.stop();
    setRecording(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const instance = new MediaRecorder(stream, { mimeType: mime });
      chunks.current = [];
      recorder.current = instance;

      instance.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.current.push(event.data);
      };

      instance.onstop = async () => {
        for (const track of stream.getTracks()) track.stop();
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        if (blob.size === 0) return;

        setTranscribing(true);
        try {
          const result = await api.transcribe(blob);
          if (result.text) onTranscript(result.text);
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : "Transcription failed.");
        } finally {
          setTranscribing(false);
        }
      };

      instance.start();
      setRecording(true);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? `Microphone unavailable: ${cause.message}`
          : "Microphone unavailable.",
      );
      setRecording(false);
    }
  }, [onTranscript]);

  const toggle = useCallback(async () => {
    if (recording) stop();
    else await start();
  }, [recording, start, stop]);

  return { recording, transcribing, error, supported, toggle };
}

/**
 * Read text aloud with the browser's speech synthesiser.
 *
 * Replaces the old server-side gTTS script: no round trip, no ffmpeg, no
 * temporary wav files, and it works offline.
 */
export function useSpeech() {
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  const supported = typeof window !== "undefined" && "speechSynthesis" in window;

  const speak = useCallback(
    (id: string, text: string) => {
      if (!supported) return;
      window.speechSynthesis.cancel();
      if (speakingId === id) {
        setSpeakingId(null);
        return;
      }
      // Strip markdown syntax and citation markers so they aren't read out.
      const spoken = text
        .replace(/```[\s\S]*?```/g, " code block ")
        .replace(/\[\d{1,2}\]/g, "")
        .replace(/[*_`#>|-]/g, " ");
      const utterance = new SpeechSynthesisUtterance(spoken);
      utterance.onend = () => setSpeakingId(null);
      utterance.onerror = () => setSpeakingId(null);
      setSpeakingId(id);
      window.speechSynthesis.speak(utterance);
    },
    [speakingId, supported],
  );

  useEffect(() => {
    if (!supported) return;
    return () => window.speechSynthesis.cancel();
  }, [supported]);

  return { speak, speakingId, supported };
}
