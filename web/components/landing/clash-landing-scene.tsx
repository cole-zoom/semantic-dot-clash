"use client";

import { type KeyboardEvent } from "react";

import { GrainCanvas } from "@/components/landing/grain-canvas";

type ClashLandingSceneProps = {
  onActivate: () => void;
};

export function ClashLandingScene({ onActivate }: ClashLandingSceneProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onActivate();
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Open chat interface"
      onClick={onActivate}
      onKeyDown={handleKeyDown}
      className="relative min-h-screen cursor-pointer overflow-hidden bg-[#050608] text-white outline-none"
    >
      <GrainCanvas />

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.08),transparent_42%),linear-gradient(180deg,rgba(5,6,8,0.18),rgba(5,6,8,0.72))]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_bottom,rgba(5,6,8,0),rgba(5,6,8,0.34)_70%)]" />

      <span className="sr-only">Click anywhere to open chat.</span>
    </div>
  );
}
