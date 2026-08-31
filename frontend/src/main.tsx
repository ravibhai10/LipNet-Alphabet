import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { RouterProvider } from "react-router-dom"
import { router } from "./router"
import "./styles/global.css"
import "./styles/nav-hero.css"
import "./styles/sections.css"
import "./styles/sections2.css"
import "./styles/workspace.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
