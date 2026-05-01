"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

type TitleOverlayProps = {
  onDismiss: () => void;
};

export function TitleOverlay({ onDismiss }: TitleOverlayProps) {
  const [entered, setEntered] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const frame = requestAnimationFrame(() => setEntered(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  const dismiss = useCallback(() => {
    if (dismissed) return;
    setDismissed(true);
  }, [dismissed]);

  const handleTransitionEnd = useCallback(
    (e: React.TransitionEvent) => {
      if (e.propertyName === "opacity" && dismissed) {
        onDismiss();
      }
    },
    [dismissed, onDismiss],
  );

  return (
    <div
      ref={containerRef}
      aria-hidden={dismissed}
      onTransitionEnd={handleTransitionEnd}
      onClick={dismiss}
      className={cn(
        "title-card",
        entered && "title-card-entered",
        dismissed && "title-card-dismissed",
      )}
    >
      {/* Shared SVG defs -- filters referenced by both sides */}
      <svg className="absolute" width="0" height="0" aria-hidden="true">
        <defs>
          {/*
           * Glitch mask filter: generates turbulence noise, then composites
           * it WITH the source graphic so noise can only appear where the
           * gradient already has opacity. No leaking past the gradient boundary.
           */}
          <filter id="glitch-l" x="0" y="0" width="100%" height="100%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.03 0.08"
              numOctaves={5}
              seed={3}
              result="noise"
            />
            <feComponentTransfer in="noise" result="binary">
              <feFuncR type="discrete" tableValues="0 1 0 1 1 0 1 0 1 0" />
              <feFuncG type="discrete" tableValues="0 1 0 1 1 0 1 0 1 0" />
              <feFuncB type="discrete" tableValues="0 1 0 1 1 0 1 0 1 0" />
              <feFuncA type="discrete" tableValues="1 1 1 1 1 1 1 1 1 1" />
            </feComponentTransfer>
            <feComposite in="SourceGraphic" in2="binary" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" />
          </filter>

          <filter id="glitch-r" x="0" y="0" width="100%" height="100%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.03 0.08"
              numOctaves={5}
              seed={7}
              result="noise"
            />
            <feComponentTransfer in="noise" result="binary">
              <feFuncR type="discrete" tableValues="0 1 1 0 1 0 0 1 0 1" />
              <feFuncG type="discrete" tableValues="0 1 1 0 1 0 0 1 0 1" />
              <feFuncB type="discrete" tableValues="0 1 1 0 1 0 0 1 0 1" />
              <feFuncA type="discrete" tableValues="1 1 1 1 1 1 1 1 1 1" />
            </feComponentTransfer>
            <feComposite in="SourceGraphic" in2="binary" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" />
          </filter>
        </defs>
      </svg>

      {/* Left side */}
      <div
        className="absolute top-0 bottom-0 left-0"
        style={{ width: "55vw" }}
      >
        <div
          className="h-full w-full bg-[#050608]"
          style={{
            maskImage: "linear-gradient(to right, black 0%, black 20%, transparent 65%)",
            WebkitMaskImage: "linear-gradient(to right, black 0%, black 20%, transparent 65%)",
            filter: "url(#glitch-l)",
          }}
        />
      </div>

      {/* Right side -- mirrored */}
      <div
        className="absolute top-0 bottom-0 right-0"
        style={{ width: "55vw" }}
      >
        <div
          className="h-full w-full bg-[#050608]"
          style={{
            maskImage: "linear-gradient(to left, black 0%, black 20%, transparent 65%)",
            WebkitMaskImage: "linear-gradient(to left, black 0%, black 20%, transparent 65%)",
            filter: "url(#glitch-r)",
          }}
        />
      </div>

    </div>
  );
}
