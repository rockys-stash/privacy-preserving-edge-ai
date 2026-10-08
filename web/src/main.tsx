import "@fontsource/geist-sans/400.css";
import "@fontsource/geist-sans/500.css";
import "@fontsource/geist-sans/600.css";
import "@fontsource/geist-mono/400.css";
import "@fontsource/geist-mono/500.css";
import "./styles.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Experiments } from "./pages/Experiments";
import { Frontier } from "./pages/Frontier";
import { Grid } from "./pages/Grid";
import { Lab } from "./pages/Lab";
import { Method } from "./pages/Method";
import { NotFound } from "./pages/NotFound";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root element");

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/lab" replace />} />
          <Route path="lab" element={<Lab />} />
          <Route path="frontier" element={<Frontier />} />
          <Route path="grid" element={<Grid />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="experiments/:exp" element={<Experiments />} />
          <Route path="method" element={<Method />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
