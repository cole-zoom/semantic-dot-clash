"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ChatOverlayShell } from "@/components/landing/chat-overlay-shell";
import { ClashLandingScene } from "@/components/landing/clash-landing-scene";
import { TitleOverlay } from "@/components/landing/title-overlay";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SESSION_STORAGE_KEY = "semantic-dot-clash-chat";

type Card = {
  id: number;
  name: string;
  rarity?: string | null;
  type?: string | null;
  elixir?: number | null;
  description?: string | null;
  role_tags?: string[] | null;
  vibe_tags?: string[] | null;
  has_evolution?: boolean | null;
  max_evolution_level?: number | null;
  evolution_image_url?: string | null;
  has_hero?: boolean | null;
  hero_image_url?: string | null;
  image_data_url?: string | null;
};

type ChatMessageResponse = {
  session_id: string;
  response_markdown: string;
  avg_elixir: number;
  cards: Card[];
  tower_card?: Card | null;
  turn_index: number;
  used_summary: boolean;
};

type ChatSessionStateResponse = {
  session_id: string;
  avg_elixir: number;
  cards: Card[];
  tower_card?: Card | null;
  turn_count: number;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  cards?: Card[];
  towerCard?: Card | null;
  avgElixir?: number;
};

const samplePrompts = [
  "Fast cycle deck that punishes Golem pushes",
  "Control deck under 3.3 elixir with strong air defense",
  "Bridge spam deck that feels off-meta and spicy",
  "A wholesome, low-toxicity deck for ladder",
];

function createSessionId(): string {
  return crypto.randomUUID();
}

function stripCardImages(card?: Card | null): Card | null | undefined {
  if (!card) return card;
  return {
    ...card,
    image_data_url: null,
  };
}

function sanitizeMessagesForStorage(messages: Message[]): Message[] {
  return messages.map((message) => ({
    ...message,
    cards: message.cards?.map((card) => stripCardImages(card) as Card),
    towerCard: stripCardImages(message.towerCard) as Card | null | undefined,
  }));
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [hasRestoredSession, setHasRestoredSession] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [showTitleCard, setShowTitleCard] = useState(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const latestDeck = useMemo(() => {
    return [...messages].reverse().find((msg) => msg.role === "assistant");
  }, [messages]);

  useEffect(() => {
    const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) {
      setSessionId(createSessionId());
      setHasRestoredSession(true);
      return;
    }

    try {
      const parsed = JSON.parse(raw) as {
        sessionId?: string;
        messages?: Message[];
      };
      setSessionId(parsed.sessionId || createSessionId());
      setMessages(Array.isArray(parsed.messages) ? parsed.messages : []);
    } catch {
      window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
      setSessionId(createSessionId());
    } finally {
      setHasRestoredSession(true);
    }
  }, []);

  useEffect(() => {
    if (!hasRestoredSession || !sessionId) return;

    window.sessionStorage.setItem(
      SESSION_STORAGE_KEY,
      JSON.stringify({
        sessionId,
        messages: sanitizeMessagesForStorage(messages),
      })
    );
  }, [hasRestoredSession, messages, sessionId]);

  useEffect(() => {
    if (!hasRestoredSession || !sessionId || messages.length === 0) return;

    let cancelled = false;

    async function hydrateLatestDeck() {
      try {
        const response = await fetch(
          `${API_URL}/api/chat/session/${encodeURIComponent(sessionId)}`
        );

        if (response.status === 404) {
          if (cancelled) return;
          window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
          setMessages([]);
          setSessionId(createSessionId());
          return;
        }

        if (!response.ok) return;

        const data = (await response.json()) as ChatSessionStateResponse;
        if (cancelled) return;

        setMessages((prev) => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i -= 1) {
            if (next[i].role === "assistant") {
              next[i] = {
                ...next[i],
                cards: data.cards,
                towerCard: data.tower_card,
                avgElixir: data.avg_elixir,
              };
              break;
            }
          }
          return next;
        });
      } catch {
        // Leave the session restored from sessionStorage even if rehydration fails.
      }
    }

    void hydrateLatestDeck();
    return () => {
      cancelled = true;
    };
  }, [hasRestoredSession, messages.length, sessionId]);

  useEffect(() => {
    if (hasRestoredSession && messages.length > 0) {
      setChatOpen(true);
    }
  }, [hasRestoredSession, messages.length]);

  const openChat = () => {
    setShowTitleCard(false);
    setChatOpen(true);
    window.requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  };

  const handleSend = async () => {
    const message = input.trim();
    if (!message || isLoading || !sessionId) return;

    if (!chatOpen) {
      setChatOpen(true);
    }

    const newMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
    };

    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Deck build failed");
      }

      const data = (await response.json()) as ChatMessageResponse;

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.response_markdown,
        cards: data.cards,
        towerCard: data.tower_card,
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

  const handleRestart = async () => {
    if (isLoading) return;

    const oldSessionId = sessionId;
    const nextSessionId = createSessionId();

    setMessages([]);
    setInput("");
    setSessionId(nextSessionId);
    window.sessionStorage.removeItem(SESSION_STORAGE_KEY);

    if (!oldSessionId) return;

    try {
      await fetch(`${API_URL}/api/chat/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: oldSessionId }),
      });
    } catch {
      // Local reset is enough to satisfy the ephemeral UX if the server call fails.
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#050608]">
      <ClashLandingScene onActivate={openChat} />

      {showTitleCard && (
        <TitleOverlay onDismiss={openChat} />
      )}

      <ChatOverlayShell isOpen={chatOpen}>
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 text-ink">
          <header className="flex flex-col gap-5 rounded-[28px] border border-white/35 bg-white/70 p-5 shadow-[0_18px_50px_rgba(0,0,0,0.12)] backdrop-blur-md md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-[18px] border border-black/10 bg-white/85 shadow-[0_12px_24px_rgba(0,0,0,0.12)]">
                <Image src="/logo.png" alt="Clash Royale" width={44} height={44} />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.32em] text-pencil">
                  Semantic Dot Clash
                </p>
                <h2 className="font-display text-3xl leading-none">
                  The war room is live.
                </h2>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-sm font-semibold">
              {["Live Arena", "Cursor Follow", "Deck Builder"].map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-black/10 bg-white/75 px-4 py-2 text-ink/80 shadow-[0_8px_20px_rgba(0,0,0,0.08)]"
                >
                  {item}
                </span>
              ))}
            </div>
          </header>

          <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
            <div className="flex min-w-0 flex-col gap-6">
              <div className="rounded-[28px] border border-white/30 bg-white/70 p-6 shadow-[0_18px_50px_rgba(0,0,0,0.12)] backdrop-blur-md">
                <div className="flex flex-wrap items-center gap-3">
                  <Badge variant="accent">Arena Overlay</Badge>
                  <Badge>Powered by Lance</Badge>
                  <Badge>Image + Text Embeddings</Badge>
                </div>
                <h2 className="mt-4 font-display text-4xl leading-tight">
                  Ask for a deck. Keep the battlefield in view.
                </h2>
                <p className="mt-3 max-w-2xl text-base text-pencil">
                  Type a vibe, strategy, or counter matchup. The agent builds an
                  eight-card battle deck plus tower troop, scores it, and
                  returns a markdown briefing with card art while the arena
                  stays alive underneath.
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

              <div className="rounded-[28px] border border-white/30 bg-white/76 p-6 shadow-[0_18px_50px_rgba(0,0,0,0.12)] backdrop-blur-md">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h3 className="font-display text-2xl">Chat Console</h3>
                    <p className="mt-1 text-sm text-pencil">
                      Clicked in. Build requests now stream through the overlay.
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs uppercase tracking-[0.35em] text-pencil">
                      Live Build
                    </span>
                    <Button
                      variant="ghost"
                      onClick={handleRestart}
                      disabled={isLoading}
                    >
                      Restart Chat
                    </Button>
                  </div>
                </div>

                <div className="soft-scrollbar mt-5 max-h-[min(42vh,420px)] space-y-4 overflow-y-auto pr-2">
                  {messages.length === 0 && (
                    <div className="rounded-2xl border-2 border-dashed border-line/20 bg-white/55 p-6 text-sm text-pencil">
                      No messages yet. Ask for a deck to get started.
                    </div>
                  )}

                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={cn(
                        "space-y-3 rounded-[22px] border p-4 shadow-[0_10px_30px_rgba(0,0,0,0.08)]",
                        message.role === "user"
                          ? "border-accent/10 bg-accentSoft/70"
                          : "border-white/50 bg-white/85"
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
                    <div className="rounded-[22px] border border-white/50 bg-white/85 p-4 shadow-[0_10px_30px_rgba(0,0,0,0.08)]">
                      <p className="text-sm text-pencil">Sketching your deck...</p>
                    </div>
                  )}
                </div>

                <div className="mt-5 flex flex-col gap-3">
                  <Textarea
                    ref={textareaRef}
                    placeholder="Describe the deck you want..."
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    className="bg-white/88"
                  />
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs text-pencil">
                      Powered by the chat agent and LanceDB embeddings.
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

            <aside className="flex min-w-0 flex-col gap-6">
              <div className="rounded-[28px] border border-white/30 bg-white/76 p-6 shadow-[0_18px_50px_rgba(0,0,0,0.12)] backdrop-blur-md">
                <h3 className="font-display text-2xl">Deckboard</h3>
                <p className="mt-2 text-sm text-pencil">
                  Latest build with art pulled directly from the Lance table.
                </p>
                {latestDeck?.cards?.length ? (
                  <div className="mt-5 grid gap-4">
                    {latestDeck.towerCard ? (
                      <div className="sketch-card bg-accentSoft/40">
                        <div className="flex items-center justify-between">
                          <h4 className="font-display text-lg">
                            Tower Troop: {latestDeck.towerCard.name}
                          </h4>
                          <Badge variant="accent">
                            {latestDeck.towerCard.elixir ?? "?"}
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs uppercase tracking-[0.2em] text-pencil">
                          {latestDeck.towerCard.rarity || "Unknown"} ·{" "}
                          {latestDeck.towerCard.type || "Card"}
                        </p>
                      </div>
                    ) : null}
                    {latestDeck.cards.map((card) => (
                      <div key={card.id} className="sketch-card">
                        <div className="relative h-24 w-full overflow-hidden rounded-md border-2 border-line bg-white">
                          {card.image_data_url ? (
                            <Image
                              src={card.image_data_url}
                              alt={card.name}
                              fill
                              className="object-cover"
                              sizes="(max-width: 1280px) 100vw, 320px"
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
                  <div className="mt-6 rounded-2xl border-2 border-dashed border-line/20 bg-white/55 p-6 text-sm text-pencil">
                    No deck yet. Send a chat request to populate this board.
                  </div>
                )}
              </div>

              <div className="rounded-[28px] border border-white/30 bg-white/76 p-6 shadow-[0_18px_50px_rgba(0,0,0,0.12)] backdrop-blur-md">
                <h3 className="font-display text-2xl">Quick Notes</h3>
                <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-pencil">
                  <li>8 cards + tower troop logic enforced by the agent.</li>
                  <li>Embeddings blend text + card art for better vibe matches.</li>
                  <li>Images are streamed as base64 from LanceDB.</li>
                </ul>
              </div>
            </aside>
          </section>
        </div>
      </ChatOverlayShell>
    </main>
  );
}
