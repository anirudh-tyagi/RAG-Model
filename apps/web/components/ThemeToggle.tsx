"use client";

import { Button } from "@/components/ui/Button";
import { MoonIcon, SunIcon } from "@/components/ui/Icons";
import { useEffect, useState } from "react";

/**
 * Reads the theme the inline bootstrap script already applied, so the button
 * never disagrees with what's on screen.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const current = document.documentElement.getAttribute("data-theme");
    setTheme(current === "dark" ? "dark" : "light");
    setMounted(true);
  }, []);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Private browsing: the choice just won't persist.
    }
    setTheme(next);
  };

  return (
    <Button
      size="icon"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      {/* Render nothing until mounted, so SSR markup matches the client. */}
      {mounted ? theme === "dark" ? <SunIcon /> : <MoonIcon /> : <span className="h-4 w-4" />}
    </Button>
  );
}
