# LipNet AI frontend redesign

Current presentation: `/` now contains six story scenes and a simple live demo, with diagnostics collapsed. `/research` preserves the detailed exhibits described below; `/demo` retains the technical tools. The story source is `frontend/src/story/StoryExperience.tsx` with scoped `styles/story.css`. The original charcoal, ivory, and mint identity is retained. See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the latest implementation and browser checks.

Navigation/motion follow-up: `story/StoryNavigation.tsx` adds visible Live Demo and Science Lab header links, a responsive All pages directory containing all 12 routes, and four destination links at the end of the story. The full live demo also has a visible link beside the inline camera section. Science Lab and Animated research are linked in the workspace sidebar. `story/StoryHeroVisual.tsx` reuses the original `visualizations/HeroVisual.tsx`: landmark face, bounded particles, mouth outline, and motion paths. GSAP entrance/parallax, staggered frames, scroll-drawn processing line, and journey reveals restore the earlier motion style. A pause button, offscreen/tab pausing, reduced-motion handling, and GSAP cleanup are retained. The shared particle loop now prevents duplicate animation-frame loops.

Follow-up: [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md) documents the newer 29-frame, activity-gated live buffer, EMA confirmation, aligned crops, and the separate AV-HuBERT training workflow. The verification below describes the original redesign milestone.

Implemented locally on Windows. The existing React 18 / TypeScript / Vite application and React Router routes are retained. No package upgrades, new packages, remote deployment, training runs, or dataset preparation were performed.

## Open the application

- Live studio: http://127.0.0.1:5173/demo/live
- Overview: http://127.0.0.1:5173/demo/overview
- Original landing page: http://127.0.0.1:5173/
- Backend health: http://127.0.0.1:8000/health

The Vite and CPU backend servers were started in hidden processes for this task and left running as requested. Logs are in the ignored .qa directory.

From the repository directory that contains app, models, and frontend, start the backend in one PowerShell terminal:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

Start the frontend in a second terminal:

~~~powershell
Set-Location frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
~~~

Dependencies were already installed. To configure a different backend, copy frontend/.env.example to frontend/.env.local and set VITE_API_BASE_URL. A restart of Vite is required after changing its environment.

## Why the old live page appeared

The application had separate landing-page and /demo/live implementations. The workspace live component referenced layout classes that had no matching styles, while workspace.css primarily styled a different version of the demo. The research workspace also omitted BackendProvider, so its consumers received the context's offline defaults. A separate status panel requested camera access when mounted, independently of the live demo.

The existing Vite server was serving this repository's frontend; a framework migration or a second replacement app was unnecessary. /demo/live now renders the redesigned implementation. /demo redirects to the new overview, and landing-page live-studio actions link directly to /demo/live.

## Routes

| Route | Result |
| --- | --- |
| / | Original first-page identity retained; supplemental sections use CPU-friendly architecture illustrations and real connection states |
| /demo/overview | New system overview, actual alphabet dataset metadata, checkpoint state, pipeline reference |
| /demo/live | New camera studio, real temporal crops, alphabet results, telemetry, local analysis, in-memory session history |
| /demo/dataset | Real letter collection and recordings, filtering, video previews, explicit validation/preparation actions |
| /demo/training | Actual precheck, validated hyperparameters, job polling, honest cancellation behavior |
| /demo/testing | Live observations and a response distribution; no accuracy or confusion matrix without labels |
| /demo/models | Search/filter and inspect actual checkpoint metadata |
| /demo/experiments | Actual saved training curves and metrics, checkpoint comparison |
| /demo/pipeline | Interactive architecture reference with implemented/unverified/unavailable scope |
| /demo/docs | Camera, connection, model-scope, and privacy guide |

## Design and motion

The first-page palette is the source of truth: charcoal #0a0a0c, warm ivory #edebe4, and mint #8fdfcb. The global palette restores missing ink and border tokens, and the workspace uses scoped CSS custom properties for surfaces, borders, status colors, spacing, radius, shadows, and transition durations.

The UI uses the installed Fraunces, Inter, and JetBrains Mono font assets. It adds an original waveform brand mark, consistent SVG icons, layered cards, camera corner markers, responsive navigation, visible keyboard focus, and meaningful empty/error states.

Motion is limited to short page entrances, button feedback, hover effects, and processing indicators tied to actual requests. Reduced-motion preferences disable workspace motion. The landing-page GSAP/SVG presentation retains its existing reduced-motion support. Supplemental continuous WebGL scenes were replaced with lightweight SVG/CSS architecture views.

The landing bundle is lazy-loaded separately from the workspace. The main production JavaScript bundle decreased from approximately 1,022 kB to approximately 332 kB before gzip.

## Camera and API integration

- Permission is requested only by Start camera.
- Pending permission requests cannot revive a cancelled or unmounted camera.
- Duplicate starts share the pending request.
- Stop and route unmount release tracks, abort inference, clear frame buffers, and clear session results.
- Device selection appears when multiple browser cameras are enumerated.
- MediaPipe uses the CPU delegate and the installed local WASM runtime copied into public/mediapipe/wasm. Fonts, WASM, and the face model require no runtime CDN download.
- FaceMesh utilities now receive the full landmark array. Mouth crops are square in video pixels and clamped to the actual video bounds.
- Detection is bounded to approximately 10 Hz, crop capture to approximately 7.7 Hz, UI readouts to 4 Hz, and inference requests to at most one every three seconds with one in flight.
- Capturing works independently of whether a visible appearance canvas exists.
- The live client uses temporal HTTP inference; the legacy unimplemented WebSocket protocol is not used by the studio.
- API health, loaded checkpoint, browser inference readiness, camera state, and local face detection are separate.
- HTTP errors, invalid JSON, timeouts, cancellation, and connection failures are handled explicitly.
- No predictions, confidence values, latency, landmarks, training percentages, global recognition events, or ablation accuracy are generated for presentation.

## Model status and limitations

The saved models/alphabet_classifier.pt is a compatible 26-class CNN-BiLSTM checkpoint. The new read-only CPU adapter loads its state dictionary strictly with weights_only=True, resamples temporal JPEG crops to the checkpoint's 30 x 140 x 46 grayscale input, and returns its actual softmax scores, top three candidates, entropy, and measured inference latency.

This is experimental alphabet classification. The saved training report records 20% best validation accuracy and 12% test accuracy on its original data. These values are shown only as saved report metrics in Experiments. They are not webcam accuracy, and browser crop accuracy has not been validated. Confidence is uncalibrated.

A compatible sentence model and decoder are not connected to the browser interface; sentence inference remains unavailable.

The current alphabet recording service reports zero recordings, speakers, and recorded letters at its configured alphabet_dataset/recordings path. Existing LRW data and the trained alphabet checkpoint are separate resources and were not moved or modified.

Running training cannot be safely interrupted by the current training service. Cancellation of a running job therefore reports this limitation rather than claiming success. No training run was launched during verification.

## Verification

- Production build and strict TypeScript check: npm run build — passed.
- No lint script or lint configuration is provided by this frontend.
- Backend: 114 tests passed, 2 skipped across CPU/runtime, camera, inference, browser adapter, services, live pipeline, and temporal buffer suites.
- Browser: 41 checks passed using the installed Chrome browser and native DevTools protocol at 1366 x 768, 900 x 1024, and 390 x 844.
- All nine workspace pages rendered without horizontal overflow or camera prompts on load.
- Camera denial, synthetic stream start/stop, cancelled late permission response, device disconnection during startup, route-unmount cleanup, offline backend, API 503/404, invalid JSON, timeout, and reduced-motion behavior passed.
- Both desktop and mobile landing pages retained the palette and contained no fabricated global events or estimated ablation accuracy.
- No uncaught browser exceptions or failed resource responses in the completed run.
- Actual HTTP inference returned 200 with three genuine model candidates from a synthetic JPEG sequence. This checks transport/model execution, not lip-reading accuracy.
- Physical camera permission and real-person prediction accuracy were not tested.
- Browser report and screenshots: .qa/browser-report.json and .qa/*.png. This directory is ignored.
- Repeat browser QA with node scripts/frontend-qa.mjs. It uses an installed Chrome/Edge browser, or CHROME_PATH, without installing dependencies.

## Main changed files

| Area | Files |
| --- | --- |
| Routing and shell | frontend/src/router.tsx; workspace/WorkspaceShell.tsx; WorkspaceSidebar.tsx; WorkspaceTopbar.tsx; StatusPanel.tsx; Icon.tsx |
| Design | styles/global.css; workspace.css; live-workspace.css; sections2.css; public/favicon.svg; index.html |
| Pages | workspace/pages/Overview.tsx; LiveDemo.tsx; Dataset.tsx; Training.tsx; Testing.tsx; Models.tsx; Experiments.tsx; Pipeline.tsx; Documentation.tsx |
| Landing integration | components/Nav.tsx; sections/Hero.tsx; LiveDemoSection.tsx; PredictionPlayground.tsx; FaceScanner.tsx; ConformerExplorer.tsx; GlobalMap.tsx; AblationStudio.tsx; Results.tsx |
| Camera | live-demo/useWebcam.ts; useLandmarker.ts; useDemoPipeline.ts; mouthUtils.ts |
| API | lib/api.ts; hooks/useBackend.tsx; workspace/api.ts; app/main.py; app/services/browser_inference.py |
| Validation | tests/test_browser_inference.py; scripts/frontend-qa.mjs |
| Local runtime | frontend/public/mediapipe/wasm; frontend/.env.example |

The original frontend was copied to a temporary backup before edits. app/main.before-premium.py.bak retains the backend before the integration changes. Existing user changes, datasets, and checkpoint files were preserved.

