/* TruthLens frontend logic */

(function () {
  "use strict";

  const SAMPLES = [
    "In a televised address on Tuesday evening, the finance ministry confirmed that the national deficit had narrowed for the third consecutive quarter, citing audited figures released by the treasury. Analysts had expected the improvement, though the pace was slower than the projection published in the spring budget. The treasury said revenue from customs duties rose by roughly four percent, offsetting weaker receipts from corporate filings. Opposition members questioned whether the accounting changes announced last year had contributed to the apparent gain, and asked for the underlying quarterly statements to be released. The finance minister said the full statements would appear on the public data portal within two weeks. Economists at two institutions said the trend was encouraging but cautioned that the coming year would bring heavier refinancing costs. The central bank has kept its policy rate unchanged since February.",
    "Scientists confirm Earth is definitively flat and every round photo from the space station was staged, according to a viral post that has gathered more than four million shares in three days. The post claims a whistleblower engineer proved the curvature models were invented to mislead the public, and it urges followers to ignore weather satellites because clouds prove the surface is a disc. Several prominent accounts amplified the claim after an image of a blue horizon was reposted without context. The image turned out to be a stock photograph of a lake in northern Finland. Fact checkers noted that the same graphic has circulated in six languages since 2019. Geographers pointed out that the horizon is visible from high altitude aircraft at ordinary commercial cruising speed, which would not happen on a flat surface. The original post has not been retracted.",
    "The transport authority said all scheduled services on the coastal line would return to normal by Friday after engineers finished replacing damaged track near the estuary. Commuters faced delays of up to forty minutes on Tuesday while crews worked overnight on the junction outside the harbour. The agency said the inspection found no structural concerns with the bridge itself. It added that a full timetable review would be published next month, including the additional evening services promised during the spring works. Local businesses near the station reported a noticeable drop in trade during the closure and asked for compensation support. A spokesperson for the mayor's office said discussions with the authority were continuing and that a review of the compensation scheme was expected before the end of the month."
  ];

  const $ = (id) => document.getElementById(id);

  const els = {
    tabs: Array.from(document.querySelectorAll(".tab")),
    panels: { text: $("panel-text"), url: $("panel-url") },
    text: $("newsText"),
    url: $("newsUrl"),
    wordCount: $("wordCount"),
    clear: $("clearText"),
    sample: document.querySelector(".sample-btn"),
    btn: $("analyzeBtn"),
    error: $("errorBox"),
    errorText: $("errorText"),
    article: $("articleBox"),
    articleTitle: $("articleTitle"),
    articleSource: $("articleSource"),
    articleWords: $("articleWords"),
    articleLink: $("articleLink"),
    articleBody: $("articleBody"),
    result: $("resultBox"),
    verdict: $("verdictBadge"),
    verdictLabel: $("verdictLabel"),
    verdictTag: $("verdictTag"),
    confValue: $("confidenceValue"),
    confFill: $("confidenceFill"),
    realFill: $("realFill"),
    fakeFill: $("fakeFill"),
    realValue: $("realValue"),
    fakeValue: $("fakeValue"),
    status: $("modelStatus"),
    themeToggle: $("themeToggle"),
  };

  let mode = "text";

  /* ---------- theme ---------- */
  const savedTheme = localStorage.getItem("truthlens-theme");
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  document.documentElement.dataset.theme = savedTheme || (prefersLight ? "light" : "dark");

  els.themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("truthlens-theme", next);
  });

  /* ---------- tabs ---------- */
  function setMode(next) {
    mode = next;
    els.tabs.forEach((tab) => {
      const active = tab.dataset.mode === next;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    Object.entries(els.panels).forEach(([key, panel]) => {
      const active = key === next;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
    hideError();
    if (next === "url") els.url.focus();
    else els.text.focus();
  }

  els.tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));

  /* ---------- word counter ---------- */
  function updateCount() {
    const words = els.text.value.trim() ? els.text.value.trim().split(/\s+/).length : 0;
    els.wordCount.textContent = words + (words === 1 ? " word" : " words");
    els.wordCount.classList.toggle("low", words > 0 && words < 20);
    els.wordCount.classList.toggle("ok", words >= 20);
  }

  els.text.addEventListener("input", updateCount);
  els.clear.addEventListener("click", () => {
    els.text.value = "";
    updateCount();
    els.text.focus();
  });

  let sampleIndex = 0;
  els.sample.addEventListener("click", () => {
    setMode("text");
    els.text.value = SAMPLES[sampleIndex % SAMPLES.length];
    sampleIndex += 1;
    updateCount();
  });

  /* ---------- helpers ---------- */
  function showError(msg) {
    els.errorText.textContent = msg;
    els.error.hidden = false;
    els.error.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideError() {
    els.error.hidden = true;
  }

  function setLoading(on) {
    els.btn.disabled = on;
    els.btn.classList.toggle("is-loading", on);
  }

  function pct(value) {
    return (Math.round(value * 1000) / 10).toFixed(1) + "%";
  }

  function readJson(res) {
    return res.json().catch(() => ({}));
  }

  function describeError(res, data) {
    if (res.status === 404) {
      return (
        (data.error || "Endpoint not found") +
        " (404). The server may be running on a different port than the page — check the URL in the terminal where you started app.py."
      );
    }
    return (data.error || "Request failed") + " (" + res.status + ").";
  }

  /* ---------- render ---------- */
  function renderResult(result) {
    els.result.hidden = false;
    els.verdict.dataset.label = result.label;
    els.verdictLabel.textContent = result.label;
    els.verdictTag.textContent = result.is_fake
      ? "This text shows patterns typical of false reporting"
      : "This text looks consistent with factual reporting";

    els.confValue.textContent = pct(result.confidence);
    els.confFill.style.width = pct(result.confidence);

    const real = result.scores.real;
    const fake = result.scores.fake;
    els.realFill.style.width = pct(real);
    els.fakeFill.style.width = pct(fake);
    els.realValue.textContent = pct(real);
    els.fakeValue.textContent = pct(fake);
  }

  function renderArticle(article) {
    els.article.hidden = false;
    els.articleTitle.textContent = article.title || "Untitled article";
    els.articleSource.textContent = article.source || "unknown source";
    els.articleWords.textContent = article.word_count + " words extracted";
    els.articleLink.href = article.url;
    els.articleBody.innerHTML = "";
    (article.paragraphs || []).forEach((p) => {
      const node = document.createElement("p");
      node.textContent = p;
      els.articleBody.appendChild(node);
    });
  }

  /* ---------- request ---------- */
  async function analyze() {
    hideError();
    els.result.hidden = true;
    els.article.hidden = true;

    const endpoint = mode === "url" ? "/api/analyze-url" : "/api/predict";
    const payload = mode === "url" ? { url: els.url.value.trim() } : { text: els.text.value };

    if (!Object.values(payload)[0]) {
      showError(mode === "url" ? "Enter an article URL first." : "Paste some article text first.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await readJson(res);
      if (!res.ok) {
        showError(describeError(res, data));
        return;
      }
      if (data.article) renderArticle(data.article);
      renderResult(data.result);
      els.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showError("Could not reach the API: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  els.btn.addEventListener("click", analyze);

  els.text.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
  });
  els.url.addEventListener("keydown", (e) => {
    if (e.key === "Enter") analyze();
  });

  /* ---------- health ---------- */
  async function checkHealth() {
    const setState = (state, text) => {
      els.status.dataset.state = state;
      els.status.querySelector(".status-text").textContent = text;
    };
    try {
      const res = await fetch("/api/health");
      const data = await readJson(res);
      if (res.status === 404) {
        setState("error", "Wrong port — API not found");
        return;
      }
      if (!res.ok) {
        setState("error", "API error " + res.status);
        return;
      }
      if (data.model_loaded) {
        setState("ready", "Model ready");
      } else {
        setState("error", "Model missing — run python train.py");
      }
    } catch (e) {
      setState("error", "API unreachable");
    }
  }

  updateCount();
  checkHealth();
})();