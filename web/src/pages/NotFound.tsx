import { Link } from "react-router-dom";
import { PageHead } from "../components/ui";

export function NotFound() {
  return (
    <>
      <PageHead title="Page not found" />
      <p className="secondary">
        There is nothing at this address. Go to the <Link to="/lab">Lab</Link>.
      </p>
    </>
  );
}
