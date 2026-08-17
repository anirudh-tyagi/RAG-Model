"use client";

import { Button, Spinner } from "@/components/ui/Button";
import { MicIcon, SendIcon, StopIcon } from "@/components/ui/Icons";
import { cn } from "@/lib/cn";
import { useVoice } from "@/lib/useVoice";
import { type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
  placeholder: string;
}

const MAX_ROWS_PX = 160;

export function Composer({ onSend, onStop, streaming, disabled, placeholder }: Props) {
  const [value, setValue] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);

  // Transcribed speech lands in the box for review rather than being sent
  // straight off, so a misheard word can be fixed before it costs a query.
  const handleTranscript = useCallback((text: string) => {
    setValue((current) => (current ? `${current} ${text}` : text));
    textarea.current?.focus();
  }, []);

  const voice = useVoice(handleTranscript);

  // Grow the textarea to fit its content, up to a cap. `value` is the trigger,
  // not an input: the height is measured from the DOM after the text changes.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run on text change
  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_ROWS_PX)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled || streaming) return;
    onSend(text);
    setValue("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter adds a newline.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border bg-bg px-3 py-3 md:px-5">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-panel p-2 focus-within:border-accent">
          <textarea
            ref={textarea}
            rows={1}
            value={value}
            disabled={disabled}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-faint disabled:opacity-60"
          />

          {voice.supported && (
            <Button
              size="icon"
              onClick={voice.toggle}
              disabled={disabled || voice.transcribing}
              aria-label={voice.recording ? "Stop recording" : "Record a question"}
              title={voice.recording ? "Stop recording" : "Record a question"}
              className={cn(voice.recording && "text-danger")}
            >
              {voice.transcribing ? (
                <Spinner />
              ) : voice.recording ? (
                <StopIcon className="h-4 w-4" />
              ) : (
                <MicIcon />
              )}
            </Button>
          )}

          {streaming ? (
            <Button
              size="icon"
              variant="primary"
              onClick={onStop}
              aria-label="Stop generating"
              title="Stop generating"
            >
              <StopIcon className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button
              size="icon"
              variant="primary"
              onClick={submit}
              disabled={disabled || !value.trim()}
              aria-label="Send"
              title="Send"
            >
              <SendIcon className="h-4 w-4" />
            </Button>
          )}
        </div>

        <p className="mt-1.5 min-h-4 px-1 text-xs text-faint">
          {voice.error ? (
            <span className="text-danger">{voice.error}</span>
          ) : voice.recording ? (
            "Recording… click stop when you're done."
          ) : voice.transcribing ? (
            "Transcribing…"
          ) : (
            "Enter to send · Shift+Enter for a new line"
          )}
        </p>
      </div>
    </div>
  );
}
