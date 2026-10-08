import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

const NAV = [
  { to: "/lab", label: "Lab" },
  { to: "/frontier", label: "Frontier" },
  { to: "/grid", label: "Grid" },
  { to: "/experiments", label: "Experiments" },
  { to: "/method", label: "Method" },
];

type Theme = "system" | "light" | "dark";

function readTheme(): Theme {
  try {
    const t = localStorage.getItem("perimeter-theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("perimeter-theme", theme);
    } catch {
      /* storage unavailable: the preference lasts for this page view */
    }
  }, [theme]);
  const next: Record<Theme, Theme> = {
    system: "light",
    light: "dark",
    dark: "system",
  };
  return (
    <button
      className="btn ghost"
      type="button"
      onClick={() => setTheme(next[theme])}
      aria-label={`Theme: ${theme}. Switch to ${next[theme]}`}
    >
      Theme: {theme}
    </button>
  );
}

export function Logo() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <rect x="2.5" y="2.5" width="9" height="11" strokeDasharray="2 1.5" />
      <path d="M9.5 8h5" />
    </svg>
  );
}

export function Layout() {
  const location = useLocation();
  useEffect(() => {
    document.getElementById("main")?.focus({ preventScroll: true });
  }, [location.pathname]);
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="topbar-row">
            <NavLink
              to="/lab"
              className="brand"
              aria-label="Perimeter, edge-device lab"
            >
              <span className="brand-mark">
                <Logo />
              </span>
              <span>
                <span className="brand-name">Perimeter</span>
                <br />
                <span className="brand-sub">
                  privacy-preserving edge AI benchmark
                </span>
              </span>
            </NavLink>
            <span className="deployment">
              What crosses the device boundary, and what keeping it in costs
            </span>
            <ThemeToggle />
          </div>
          <nav className="tabs" aria-label="Sections">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) => (isActive ? "active" : "")}
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main id="main" className="main" tabIndex={-1}>
        <Outlet />
      </main>
    </>
  );
}
