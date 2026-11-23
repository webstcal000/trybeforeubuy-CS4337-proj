import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const API_TIMEOUT_MS = 60_000;

async function readError(response) {
  try {
    const body = await response.json();
    return body?.error?.message || "The request could not be completed. Please try again.";
  } catch {
    return "The service returned an unexpected response. Please try again.";
  }
}

function Icon({ name, size = 18 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  if (name === "arrow") {
    return <svg {...common}><path d="M5 12h14M13 5l7 7-7 7" /></svg>;
  }
  if (name === "upload") {
    return <svg {...common}><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v5h14v-5" /></svg>;
  }
  if (name === "download") {
    return <svg {...common}><path d="M12 4v12m0 0 4.5-4.5M12 16l-4.5-4.5M5 19h14" /></svg>;
  }
  if (name === "spark") {
    return <svg {...common}><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3ZM19 16l.7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" /></svg>;
  }
  return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 11v5m0-8h.01" /></svg>;
}

function PersonPlaceholder({ onUpload, disabled }) {
  return (
    <div className="empty-canvas">
      <div className="empty-canvas__art" aria-hidden="true">
        <div className="figure-head" />
        <div className="figure-body" />
        <div className="figure-leg figure-leg--left" />
        <div className="figure-leg figure-leg--right" />
      </div>
      <p className="eyebrow">Your fitting room</p>
      <h2>Start with a photo</h2>
      <p className="empty-canvas__copy">
        Upload a clear, front-facing photo to see how a new look might feel.
      </p>
      <button className="button button--dark" type="button" onClick={onUpload} disabled={disabled}>
        <Icon name="upload" />
        Upload your photo
      </button>
      <span className="file-note">JPEG or PNG · up to 10 MB</span>
    </div>
  );
}

function App() {
  const fileInputRef = useRef(null);
  const photoRef = useRef(null);
  const resultRef = useRef(null);
  const objectUrlsRef = useRef(new Set());
  const [health, setHealth] = useState(null);
  const [garments, setGarments] = useState([]);
  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("all");
  const [selectedId, setSelectedId] = useState(null);
  const [photo, setPhoto] = useState(null);
  const [result, setResult] = useState(null);
  const [showAfter, setShowAfter] = useState(true);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [catalogError, setCatalogError] = useState("");
  const [sampleMessage, setSampleMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const selectedGarment = useMemo(
    () => garments.find((garment) => garment.id === selectedId) || null,
    [garments, selectedId],
  );

  const visibleGarments = useMemo(
    () => category === "all"
      ? garments
      : garments.filter((garment) => garment.category === category),
    [category, garments],
  );

  const rememberUrl = useCallback((url) => {
    objectUrlsRef.current.add(url);
    return url;
  }, []);

  const releaseUrl = useCallback((url) => {
    if (url && objectUrlsRef.current.delete(url)) URL.revokeObjectURL(url);
  }, []);

  const replacePhoto = useCallback((nextPhoto) => {
    releaseUrl(photoRef.current?.src);
    photoRef.current = nextPhoto;
    setPhoto(nextPhoto);
  }, [releaseUrl]);

  const replaceResult = useCallback((nextResult) => {
    releaseUrl(resultRef.current?.src);
    resultRef.current = nextResult;
    setResult(nextResult);
  }, [releaseUrl]);

  useEffect(() => {
    let active = true;
    async function initialize() {
      const [healthResponse, catalogResponse, sampleResponse] = await Promise.allSettled([
        fetch("/api/health"),
        fetch("/api/garments"),
        fetch("/api/sample-person"),
      ]);
      if (!active) return;

      if (healthResponse.status === "fulfilled" && healthResponse.value.ok) {
        setHealth(await healthResponse.value.json());
      } else {
        setHealth({ service: "unavailable", issues: [] });
        setErrorMessage("The local service is unavailable. Start the API and refresh this page.");
      }

      if (catalogResponse.status === "fulfilled" && catalogResponse.value.ok) {
        const catalog = await catalogResponse.value.json();
        setGarments(catalog.garments);
        setCategories(catalog.categories);
      } else if (catalogResponse.status === "fulfilled") {
        setCatalogError(await readError(catalogResponse.value));
      } else {
        setCatalogError("The garment catalog could not be reached. Start the API and try again.");
      }

      if (sampleResponse.status === "fulfilled" && sampleResponse.value.ok) {
        const blob = await sampleResponse.value.blob();
        if (!active || photoRef.current) return;
        const extension = blob.type === "image/png" ? "png" : "jpg";
        const file = new File([blob], `sample-person.${extension}`, { type: blob.type });
        const src = rememberUrl(URL.createObjectURL(blob));
        replacePhoto({ file, src, label: "Sample photo", isSample: true });
      } else {
        setSampleMessage("No sample photo is installed. Upload your own to get started.");
      }
      setLoading(false);
    }

    initialize().catch(() => {
      if (!active) return;
      setLoading(false);
      setErrorMessage("The studio could not load. Check that the local API is running, then refresh.");
    });

    return () => {
      active = false;
    };
  }, [rememberUrl, replacePhoto]);

  useEffect(() => () => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current.clear();
  }, []);

  const openFilePicker = () => fileInputRef.current?.click();

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      setErrorMessage("Choose a JPEG or PNG image.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setErrorMessage("Choose an image smaller than 10 MB.");
      return;
    }
    try {
      const decoded = await createImageBitmap(file);
      const validDimensions = decoded.width >= 64 && decoded.height >= 64
        && decoded.width <= 8000 && decoded.height <= 8000
        && decoded.width * decoded.height <= 40_000_000;
      decoded.close();
      if (!validDimensions) {
        setErrorMessage("Use an image at least 64 × 64 pixels and no larger than 8,000 pixels per side or 40 megapixels.");
        return;
      }
    } catch {
      setErrorMessage("That file could not be opened as an image. Choose a valid JPEG or PNG.");
      return;
    }

    const src = rememberUrl(URL.createObjectURL(file));
    replacePhoto({ file, src, label: file.name, isSample: false });
    replaceResult(null);
    setShowAfter(true);
    setErrorMessage("");
    setSampleMessage("");
  };

  const selectGarment = (garmentId) => {
    if (generating) return;
    setSelectedId(garmentId);
    replaceResult(null);
    setShowAfter(true);
    setErrorMessage("");
  };

  const tryOn = async () => {
    if (!photo || !selectedGarment || generating) return;
    setGenerating(true);
    setErrorMessage("");
    replaceResult(null);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);
    try {
      const body = new FormData();
      body.append("person", photo.file, photo.file.name);
      body.append("garment_id", String(selectedGarment.id));
      const response = await fetch("/api/try-on", {
        method: "POST",
        body,
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await readError(response));
      const blob = await response.blob();
      if (blob.type !== "image/png") {
        throw new Error("The service returned an invalid preview. Please try again.");
      }
      const src = rememberUrl(URL.createObjectURL(blob));
      replaceResult({ src, filename: `trybeforeubuy-${selectedGarment.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png` });
      setShowAfter(true);
    } catch (error) {
      setErrorMessage(
        error.name === "AbortError"
          ? "This is taking longer than expected. The server may still be working; please retry in a moment."
          : error.message || "The preview could not be generated. Please try again.",
      );
    } finally {
      window.clearTimeout(timeout);
      setGenerating(false);
    }
  };

  const displayedSrc = result && showAfter ? result.src : photo?.src;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="TryBeforeUBuy studio">
         {/* <span className="wordmark__symbol">t<span>b</span></span> */}
          <span>TryBeforeUBuy</span>
        </a>
        <div className="topbar__right">
          <span className="topbar__edition">THE VIRTUAL FITTING ROOM</span>
          <span className={`service-indicator ${health?.pipeline_ready ? "is-ready" : ""}`}>
            <span aria-hidden="true" />
            {health?.pipeline_ready ? "Studio ready" : "Needs setup"}
          </span>
        </div>
      </header>

      <section className="intro" aria-labelledby="page-title">
        <div>
          <p className="eyebrow">A space to explore</p>
          <h1 id="page-title">Find your <em>next</em> favorite.</h1>
          <p className="intro__copy">
            A styling preview, made for curiosity. Choose a look and see it on you.
          </p>
        </div>
        <div className="intro__note">
          <Icon name="spark" size={17} />
          <span>Made for inspiration</span>
        </div>
      </section>

      {errorMessage && (
        <div className="notice notice--error" role="alert">
          <Icon name="info" />
          <p>{errorMessage}</p>
          <button type="button" className="notice__dismiss" onClick={() => setErrorMessage("")} aria-label="Dismiss message">
            ×
          </button>
        </div>
      )}

      <section className="studio" aria-label="Virtual styling studio">
        <section className="preview-panel panel" aria-labelledby="preview-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">01 / Your canvas</p>
              <h2 id="preview-title">{photo ? "The fitting room" : "Your fitting room"}</h2>
            </div>
            <div className="panel-heading__actions">
              {photo && (
                <span className="photo-name" title={photo.label}>
                  <span className="photo-name__dot" />
                  {photo.label}
                </span>
              )}
              <label className={`button button--quiet upload-control ${generating ? "is-disabled" : ""}`}>
                <Icon name="upload" size={16} />
                <span>{photo ? "Change photo" : "Upload photo"}</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  aria-label="Upload a JPEG or PNG photo"
                  disabled={generating}
                  onChange={handleUpload}
                />
              </label>
            </div>
          </div>

          <div className={`image-stage ${photo ? "image-stage--photo" : ""}`}>
            {photo ? (
              <>
                <div className="image-stage__wash" />
                <img
                  className={`person-image ${result && showAfter ? "person-image--result" : ""}`}
                  src={displayedSrc}
                  alt={result && showAfter ? `Try-on preview with ${selectedGarment?.name}` : "Original photo"}
                />
                {generating && (
                  <div className="processing-chip" role="status" aria-live="polite">
                    <span className="processing-dot" />
                    Creating your preview
                  </div>
                )}
              </>
            ) : (
              <PersonPlaceholder onUpload={openFilePicker} disabled={generating} />
            )}
            {result && photo && (
              <div className="compare-control" role="group" aria-label="Compare original photo and try-on preview">
                <button type="button" aria-pressed={!showAfter} onClick={() => setShowAfter(false)}>
                  Before
                </button>
                <button type="button" aria-pressed={showAfter} onClick={() => setShowAfter(true)}>
                  After
                </button>
              </div>
            )}
          </div>

          <div className="preview-footer">
            <div className="preview-tip">
              <Icon name="info" size={16} />
              <span>Front-facing, well-lit photos work best.</span>
            </div>
            {result && (
              <a className="download-link" href={result.src} download={result.filename}>
                <Icon name="download" size={16} />
                Download PNG
              </a>
            )}
          </div>
          {sampleMessage && !photo && <p className="inline-hint">{sampleMessage}</p>}
        </section>

        <aside className="catalog-panel panel" aria-labelledby="catalog-title">
          <div className="panel-heading catalog-heading">
            <div>
              <p className="eyebrow">02 / The edit</p>
              <h2 id="catalog-title">Choose a piece</h2>
            </div>
            {garments.length > 0 && <span className="item-count">{garments.length} PIECES</span>}
          </div>

          {categories.length > 0 && (
            <div className="filter-row" aria-label="Filter garments by category">
              <button
                type="button"
                className={`filter-chip ${category === "all" ? "is-active" : ""}`}
                aria-pressed={category === "all"}
                disabled={generating}
                onClick={() => setCategory("all")}
              >
                All pieces
              </button>
              {categories.map((item) => (
                <button
                  type="button"
                  className={`filter-chip ${category === item ? "is-active" : ""}`}
                  aria-pressed={category === item}
                  key={item}
                  disabled={generating}
                  onClick={() => setCategory(item)}
                >
                  {item.replaceAll("_", " ")}
                </button>
              ))}
            </div>
          )}

          <div className="garment-grid" aria-live="polite">
            {loading ? (
              <div className="catalog-message">
                <span className="loading-mark" aria-hidden="true" />
                <p>Opening the edit…</p>
              </div>
            ) : catalogError ? (
              <div className="catalog-message catalog-message--empty">
                <span className="catalog-message__icon"><Icon name="info" size={22} /></span>
                <h3>The wardrobe is waiting</h3>
                <p>{catalogError}</p>
                <p className="catalog-message__detail">Add the DeepFashion files to <code>downloads/</code>, then refresh.</p>
              </div>
            ) : visibleGarments.length === 0 ? (
              <div className="catalog-message catalog-message--empty">
                <span className="catalog-message__icon"><Icon name="info" size={22} /></span>
                <h3>No pieces in this edit</h3>
                <p>There are no matching garments in this category.</p>
                <button type="button" className="text-button" onClick={() => setCategory("all")}>
                  Show all pieces <Icon name="arrow" size={15} />
                </button>
              </div>
            ) : visibleGarments.map((garment) => {
              const isSelected = selectedId === garment.id;
              return (
                <button
                  type="button"
                  className={`garment-card ${isSelected ? "is-selected" : ""}`}
                  aria-pressed={isSelected}
                  key={garment.id}
                  disabled={generating}
                  onClick={() => selectGarment(garment.id)}
                >
                  <span className="garment-card__image-wrap">
                    <img src={garment.imageUrl} alt={`${garment.name} clothing item`} loading="lazy" />
                    {isSelected && <span className="selected-mark" aria-hidden="true">✓</span>}
                  </span>
                  <span className="garment-card__meta">
                    <span className="garment-card__name">{garment.name}</span>
                    <span className="garment-card__category">{garment.category.replaceAll("_", " ")}</span>
                  </span>
                  <span className="selection-label">{isSelected ? "Selected" : "Select piece"}</span>
                </button>
              );
            })}
          </div>

          <div className="tryon-bar">
            <div className="tryon-bar__selection">
              {selectedGarment ? (
                <>
                  <span className="selection-kicker">YOUR SELECTION</span>
                  <span className="selection-name">{selectedGarment.name}</span>
                </>
              ) : (
                <>
                  <span className="selection-kicker">READY WHEN YOU ARE</span>
                  <span className="selection-name selection-name--muted">Choose a piece to begin</span>
                </>
              )}
            </div>
            <button
              className="button button--accent tryon-button"
              type="button"
              disabled={!photo || !selectedGarment || generating}
              onClick={tryOn}
            >
              {generating ? (
                <>
                  <span className="button-spinner" aria-hidden="true" />
                  Working
                </>
              ) : (
                <>
                  Try it on <Icon name="arrow" size={17} />
                </>
              )}
            </button>
          </div>
          {!photo && !catalogError && (
            <p className="selection-hint">Upload a photo and select a piece to make a preview.</p>
          )}
        </aside>
      </section>

      <footer className="page-footer">
        <span>TRYBEFOREUBUY <span aria-hidden="true">·</span> A PERSONAL STYLE STUDIO</span>
        <span>Visual styling preview only — not a guarantee of fit.</span>
      </footer>
    </main>
  );
}

export default App;
