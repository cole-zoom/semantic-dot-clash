import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        paper: "var(--paper)",
        ink: "var(--ink)",
        pencil: "var(--pencil)",
        accent: "var(--accent)",
        accentSoft: "var(--accent-soft)",
        line: "var(--line)",
      },
      fontFamily: {
        display: ["var(--font-display)", "serif"],
        body: ["var(--font-body)", "serif"],
      },
      boxShadow: {
        sketch: "0.15rem 0.2rem 0 0 rgba(0,0,0,0.15)",
        float: "0 12px 40px rgba(0, 0, 0, 0.08)",
      },
      backgroundImage: {
        "paper-lines": "repeating-linear-gradient(0deg, rgba(15,15,15,0.04) 0px, rgba(15,15,15,0.04) 1px, transparent 1px, transparent 28px)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
