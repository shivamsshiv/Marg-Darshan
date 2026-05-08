import React from "react";
import { createRoot } from "react-dom/client";
import { Camera, Mic, MicOff, Navigation, Volume2, VolumeX } from "lucide-react";
import "./styles.css";

type Detection = {
  label: string;
  confidence: number;
  bbox: number[];
  track_id?: number;
  direction: "left" | "center" | "right" | "unknown";
  distance_hint: "very_near" | "near" | "mid" | "far" | "unknown";
  priority: number;
};

type GuidanceResponse = {
  status: "ok" | "error";
  mode: "navigation" | "question" | "ocr" | "idle";
  guidance: string;
  intent: { name: string; confidence: number; raw_text: string };
  detections: Detection[];
  ocr_text: string;
  spoken: boolean;
  debug: Record<string, unknown>;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const FRAME_INTERVAL_MS = 333;
const PROCESSING_WIDTH = 640;

function App() {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const latestVoiceRef = React.useRef("");
  const isSendingRef = React.useRef(false);
  const lastGuidanceRef = React.useRef("");
  const recognitionRef = React.useRef<any>(null);

  const [isRunning, setIsRunning] = React.useState(false);
  const [micEnabled, setMicEnabled] = React.useState(true);
  const [voiceOut, setVoiceOut] = React.useState(true);
  const [voiceText, setVoiceText] = React.useState("");
  const [guidance, setGuidance] = React.useState("Press Start. Camera and mic will run together.");
  const [result, setResult] = React.useState<GuidanceResponse | null>(null);
  const [error, setError] = React.useState("");
  const [permissionHelp, setPermissionHelp] = React.useState("");
  const [processingSize, setProcessingSize] = React.useState({ width: 1, height: 1 });

  const speak = React.useCallback(
    (text: string) => {
      if (!voiceOut || !text || text === lastGuidanceRef.current) return;
      lastGuidanceRef.current = text;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.02;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    },
    [voiceOut],
  );

  const startMic = React.useCallback(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition || !micEnabled) return;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-IN";
    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        transcript += event.results[i][0].transcript;
      }
      const clean = transcript.trim();
      latestVoiceRef.current = clean;
      setVoiceText(clean);
    };
    recognition.onerror = () => undefined;
    recognition.onend = () => {
      if (streamRef.current && micEnabled) {
        try {
          recognition.start();
        } catch {
          // Browser can throw if recognition is already active.
        }
      }
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      // Safe to ignore; the browser may require a user gesture.
    }
  }, [micEnabled]);

  const stopMic = React.useCallback(() => {
    recognitionRef.current?.stop?.();
    recognitionRef.current = null;
  }, []);

  const start = async () => {
    setError("");
    setPermissionHelp("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support camera access. Open the app in Chrome or Edge.");
      }

      const cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 960 }, height: { ideal: 540 } },
        audio: false,
      });

      let stream = cameraStream;
      if (micEnabled) {
        try {
          const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
          audioStream.getAudioTracks().forEach((track) => stream.addTrack(track));
        } catch {
          setPermissionHelp(
            "Camera is running, but microphone permission is blocked. Allow microphone access in site settings, then press Stop and Start again.",
          );
        }
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsRunning(true);
      startMic();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not start camera or microphone.";
      const isDenied =
        message.toLowerCase().includes("permission") ||
        message.toLowerCase().includes("denied") ||
        message.toLowerCase().includes("notallowed");
      setError(isDenied ? "Camera permission is denied." : message);
      setPermissionHelp(
        isDenied
          ? "Click the camera/lock icon near the address bar, allow Camera and Microphone for localhost:5173, reload the page, then press Start."
          : "If you are using an in-app browser, open http://localhost:5173 in Chrome or Edge because some embedded browsers block camera access.",
      );
    }
  };

  const stop = () => {
    stopMic();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsRunning(false);
    window.speechSynthesis.cancel();
  };

  const sendFrame = React.useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isSendingRef.current) return;
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) return;
    isSendingRef.current = true;
    const canvas = canvasRef.current;
    const scale = Math.min(1, PROCESSING_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    setProcessingSize({ width: canvas.width, height: canvas.height });
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.62),
    );
    if (!blob) {
      isSendingRef.current = false;
      return;
    }
    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    form.append("voice_text", latestVoiceRef.current);
    form.append("speak", "false");
    try {
      const response = await fetch(`${API_URL}/api/guidance/frame`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const data = (await response.json()) as GuidanceResponse;
      setResult(data);
      setGuidance(data.guidance);
      speak(data.guidance);
      if (latestVoiceRef.current) {
        latestVoiceRef.current = "";
        setVoiceText("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not process frame.");
    } finally {
      isSendingRef.current = false;
    }
  }, [speak]);

  React.useEffect(() => {
    if (!isRunning) return;
    const id = window.setInterval(sendFrame, FRAME_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [isRunning, sendFrame]);

  React.useEffect(() => () => stop(), []);

  const topDetections = result?.detections.filter((item) => item.priority >= 0.45).slice(0, 2) ?? [];

  return (
    <main className="app-shell">
      <section className="stage">
        <div className="video-wrap">
          <video ref={videoRef} playsInline muted aria-label="Live camera feed" />
          <canvas ref={canvasRef} hidden />
          <div className="crosshair" />
          {topDetections.map((item) => (
            <div
              className={`box box-${item.direction}`}
              key={`${item.track_id}-${item.label}-${item.direction}`}
              style={{
                left: `${(item.bbox[0] / Math.max(processingSize.width, 1)) * 100}%`,
                top: `${(item.bbox[1] / Math.max(processingSize.height, 1)) * 100}%`,
                width: `${((item.bbox[2] - item.bbox[0]) / Math.max(processingSize.width, 1)) * 100}%`,
                height: `${((item.bbox[3] - item.bbox[1]) / Math.max(processingSize.height, 1)) * 100}%`,
              }}
            >
              <span>{item.direction}</span>
            </div>
          ))}
        </div>
      </section>

      <aside className="control-panel">
        <header>
          <div className="brand-mark">
            <Navigation size={24} />
          </div>
          <div>
            <h1>AI Guidance</h1>
            <p>Camera navigation with live voice questions</p>
          </div>
        </header>

        <div className="guidance">
          <span className={`mode mode-${result?.mode ?? "idle"}`}>{result?.mode ?? "ready"}</span>
          <h2>{guidance}</h2>
          {voiceText && <p className="heard">Heard: {voiceText}</p>}
          {error && <p className="error">{error}</p>}
          {permissionHelp && <p className="permission-help">{permissionHelp}</p>}
        </div>

        <div className="actions">
          <button className="primary" onClick={isRunning ? stop : start}>
            <Camera size={18} />
            {isRunning ? "Stop" : "Start"}
          </button>
          <button
            aria-pressed={micEnabled}
            onClick={() => {
              setMicEnabled((value) => !value);
              if (micEnabled) stopMic();
            }}
          >
            {micEnabled ? <Mic size={18} /> : <MicOff size={18} />}
            Mic
          </button>
          <button aria-pressed={voiceOut} onClick={() => setVoiceOut((value) => !value)}>
            {voiceOut ? <Volume2 size={18} /> : <VolumeX size={18} />}
            Voice
          </button>
        </div>

        <div className="status-grid">
          <div>
            <span>Frame Rate</span>
            <strong>3 FPS</strong>
          </div>
          <div>
            <span>Intent</span>
            <strong>{result?.intent.name ?? "none"}</strong>
          </div>
          <div>
            <span>Objects</span>
            <strong>{topDetections.length}</strong>
          </div>
          <div>
            <span>OCR</span>
            <strong>{result?.ocr_text ? "read" : "standby"}</strong>
          </div>
        </div>

        <div className="detections">
          <h3>Live Scene</h3>
          {topDetections.length === 0 ? (
            <p>No detection yet.</p>
          ) : (
            topDetections.map((item) => (
              <article key={`${item.track_id}-${item.priority}`}>
                <strong>{item.direction}</strong>
                <span>{item.distance_hint} · {(item.confidence * 100).toFixed(0)}%</span>
              </article>
            ))
          )}
        </div>
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
