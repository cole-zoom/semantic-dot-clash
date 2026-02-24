import type { Metadata } from "next";
import { Kalam, Spectral } from "next/font/google";

import "./globals.css";

const kalam = Kalam({
  weight: ["300", "400", "700"],
  subsets: ["latin"],
  variable: "--font-display",
});

const spectral = Spectral({
  weight: ["300", "400", "600", "700"],
  subsets: ["latin"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "Semantic Dot Clash",
  description: "Sketch-styled Clash Royale deck builder.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${kalam.variable} ${spectral.variable}`}>
      <body>{children}</body>
    </html>
  );
}
