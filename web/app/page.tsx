"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Card = {
  id: number;
  name: string;
  rarity?: string | null;
  type?: string | null;
  elixir?: number | null;
  description?: string | null;
  role_tags?: string[] | null;
  vibe_tags?: string[] | null;
  image_data_url?: string | null;
};

type DeckResponse = {
  prompt: string;
  response_markdown: string;
  avg_elixir: number;
  cards: Card[];
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  cards?: Card[];
  avgElixir?: number;
};

const samplePrompts = [
  "Fast cycle deck that punishes Golem pushes",
  "Control deck under 3.3 elixir with strong air defense",
  "Bridge spam deck that feels off-meta and spicy",
  "A wholesome, low-toxicity deck for ladder",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const latestDeck = useMemo(() => {
    return [...messages].reverse().find((msg) => msg.role === "assistant");
  }, [messages]);

  const handleSend = async () => {
    const prompt = input.trim();
    if (!prompt || isLoading) return;

    const newMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: prompt,
    };

    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/deck`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Deck build failed");
      }

      const data = (await response.json()) as DeckResponse;

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.response_markdown,
        cards: data.cards,
        avgElixir: data.avg_elixir,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          error instanceof Error
            ? `**Something broke.** ${error.message}`
            : "**Something broke.** Please try again.",
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-paper text-ink">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-12 px-6 pb-16 pt-8">
        <header className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="sketch-panel flex h-14 w-14 items-center justify-center bg-white">
              <Image src="/logo.png" alt="Clash Royale" width={44} height={44} />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-pencil">
                Semantic Dot Clash
              </p>
              <h1 className="font-display text-3xl lg:text-4xl">
                Deck Builder, drawn in pencil.
              </h1>
            </div>
          </div>
          <nav className="flex flex-wrap items-center gap-3 text-sm font-semibold">
            {["Stats", "Decks", "Cards", "Meta", "Tools"].map((item) => (
              <button
                key={item}
                className="sketch-pill text-ink transition hover:-translate-y-0.5"
              >
                {item}
              </button>
            ))}
          </nav>
        </header>

        <section className="grid gap-10 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="flex flex-col gap-6">
            <div className="sketch-panel bg-white p-6">
              <div className="flex flex-wrap items-center gap-3">
                <Badge variant="accent">Sketch UI</Badge>
                <Badge>Powered by Lance</Badge>
                <Badge>Image + Text Embeddings</Badge>
              </div>
              <h2 className="mt-4 font-display text-4xl leading-tight">
                Ask for a deck. Get a hand-drawn gameplan.
              </h2>
              <p className="mt-3 text-base text-pencil">
                Type a vibe, strategy, or counter matchup. The agent builds an
                eight-card list, scores it, and brings back a markdown briefing
                plus the card art.
              </p>
              <div className="mt-5 flex flex-wrap gap-2">
                {samplePrompts.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => setInput(prompt)}
                    className="sketch-chip"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            <div className="sketch-panel bg-white p-6">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-2xl">Chat Console</h3>
                <span className="text-xs uppercase tracking-[0.35em] text-pencil">
                  Live Build
                </span>
              </div>

              <div className="mt-4 max-h-[420px] space-y-4 overflow-y-auto pr-2">
                {messages.length === 0 && (
                  <div className="rounded-lg border-2 border-dashed border-line/40 p-6 text-sm text-pencil">
                    No messages yet. Ask for a deck to get started.
                  </div>
                )}

                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={cn(
                      "sketch-panel-sm space-y-3",
                      message.role === "user"
                        ? "bg-accentSoft/70"
                        : "bg-white"
                    )}
                  >
                    <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-pencil">
                      <span>{message.role === "user" ? "You" : "Agent"}</span>
                      {message.avgElixir ? (
                        <span>Avg Elixir: {message.avgElixir.toFixed(2)}</span>
                      ) : null}
                    </div>
                    <div className="prose prose-sm max-w-none text-ink prose-headings:font-display prose-strong:text-ink">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="sketch-panel-sm bg-white">
                    <p className="text-sm text-pencil">Sketching your deck...</p>
                  </div>
                )}
              </div>

              <div className="mt-4 flex flex-col gap-3">
                <Textarea
                  placeholder="Describe the deck you want..."
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                />
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-xs text-pencil">
                    Powered by the CLI agent and LanceDB embeddings.
                  </p>
                  <Button
                    variant="accent"
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                  >
                    Send Request
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <aside className="flex flex-col gap-6">
            <div className="sketch-panel bg-white p-6">
              <h3 className="font-display text-2xl">Deckboard</h3>
              <p className="mt-2 text-sm text-pencil">
                Latest build with art pulled directly from the Lance table.
              </p>
              {latestDeck?.cards?.length ? (
                <div className="mt-5 grid gap-4">
                  {latestDeck.cards.map((card) => (
                    <div key={card.id} className="sketch-card">
                      <div className="relative h-24 w-full overflow-hidden rounded-md border-2 border-line bg-white">
                        {card.image_data_url ? (
                          <Image
                            src={card.image_data_url}
                            alt={card.name}
                            fill
                            className="object-cover"
                            sizes="(max-width: 1024px) 100vw, 300px"
                            unoptimized
                          />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center text-xs text-pencil">
                            No image
                          </div>
                        )}
                      </div>
                      <div className="mt-3 flex items-center justify-between">
                        <h4 className="font-display text-lg">{card.name}</h4>
                        <Badge variant="accent">{card.elixir ?? "?"}</Badge>
                      </div>
                      <p className="mt-1 text-xs uppercase tracking-[0.2em] text-pencil">
                        {card.rarity || "Unknown"} · {card.type || "Card"}
                      </p>
                      {card.role_tags?.length ? (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {card.role_tags.map((tag) => (
                            <Badge key={tag}>{tag}</Badge>
                          ))}
                        </div>
                      ) : null}
                      {card.vibe_tags?.length ? (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {card.vibe_tags.slice(0, 3).map((tag) => (
                            <Badge key={tag} variant="accent">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-6 rounded-lg border-2 border-dashed border-line/40 p-6 text-sm text-pencil">
                  No deck yet. Send a chat request to populate this board.
                </div>
              )}
            </div>

            <div className="sketch-panel bg-white p-6">
              <h3 className="font-display text-2xl">Quick Notes</h3>
              <ul className="mt-3 list-disc pl-5 text-sm text-pencil">
                <li>8 cards + tower troop logic enforced by the agent.</li>
                <li>Embeddings blend text + card art for better vibe matches.</li>
                <li>Images are streamed as base64 from LanceDB.</li>
              </ul>
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}
