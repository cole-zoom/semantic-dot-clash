"use client";

import { type ReactNode } from "react";

import { cn } from "@/lib/utils";

type ChatOverlayShellProps = {
  isOpen: boolean;
  children: ReactNode;
};

export function ChatOverlayShell({
  isOpen,
  children,
}: ChatOverlayShellProps) {
  return (
    <div
      aria-hidden={!isOpen}
      className={cn(
        "fixed inset-0 z-30 transition-all duration-300",
        isOpen ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
      )}
    >
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(5,6,8,0.14),rgba(5,6,8,0.58))] backdrop-blur-[3px]" />
      <div
        className={cn(
          "relative flex min-h-screen items-start justify-center overflow-y-auto p-4 transition-transform duration-300 sm:p-6 lg:p-8",
          isOpen ? "translate-y-0" : "translate-y-4"
        )}
      >
        <div className="overlay-shell w-full max-w-7xl rounded-[32px] p-4 sm:p-5 lg:p-6">
          {children}
        </div>
      </div>
    </div>
  );
}
