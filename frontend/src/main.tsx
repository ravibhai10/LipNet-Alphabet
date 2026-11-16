import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { RouterProvider } from "react-router-dom"
import { router } from "./router"
import { AppChrome } from "./ui/AppChrome"
import "./styles/global.css"
import "./styles/nav-hero.css"
import "./styles/sections.css"
import "./styles/sections2.css"
import "./styles/workspace.css"

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("/sw.js")
  })
}

// Automatically reload once if a dynamic import fails due to chunk hash mismatch on new deployment
window.addEventListener("vite:preloadError", () => {
  const key = "lipnet_chunk_reload"
  const now = Date.now()
  const last = parseInt(sessionStorage.getItem(key) || "0", 10)
  if (now - last > 8000) {
    sessionStorage.setItem(key, now.toString())
    window.location.reload()
  }
})

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppChrome>
      <RouterProvider router={router} />
    </AppChrome>
  </StrictMode>,
)
