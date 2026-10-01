/* Copyright 2026 the LlamaFactory team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Browser side of the modern LLaMA Board layout. Loaded in <head>, before Gradio mounts.
 *
 * Presentation only: pages, style / dark mode, sidebar collapse + resize, the command palette and
 * tooltips. It never writes into a Gradio component and never touches training state. */
(function () {
  "use strict";
  var root = document.documentElement;
  var PAGES = ["model", "train", "eval", "infer", "export"];
  var NAV_MIN = 208, NAV_MAX = 380, NAV_DEFAULT = 248;
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) {} }
  };

  /* ------------------------------------------------------------ responsive rules
   * The stylesheet's @media blocks are served as html[data-lf-mq~="mqN"] rules (see cssutil.py);
   * mirror each media condition into that attribute. */
  var MQ = window.LF_MQ || {};
  var mqLists = Object.keys(MQ).map(function (key) {
    return { key: key, mql: window.matchMedia ? window.matchMedia(MQ[key]) : null };
  });
  function syncMedia() {
    root.setAttribute("data-lf-mq", mqLists.filter(function (m) { return m.mql && m.mql.matches; })
      .map(function (m) { return m.key; }).join(" "));
  }
  mqLists.forEach(function (m) {
    if (!m.mql) return;
    if (m.mql.addEventListener) m.mql.addEventListener("change", syncMedia);
    else if (m.mql.addListener) m.mql.addListener(syncMedia);
  });
  syncMedia();

  /* ------------------------------------------------------------ style & mode */
  var sysDark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function currentMode() {
    var saved = store.get("lf_mode");
    if (saved === "dark" || saved === "light") return saved;
    return sysDark && sysDark.matches ? "dark" : "light";
  }
  function applyAppearance() {
    var style = store.get("lf_style") === "soft" ? "soft" : "eng";
    var mode = currentMode();
    root.setAttribute("data-lf-style", style);
    root.setAttribute("data-lf-mode", mode);
    root.classList.toggle("dark", mode === "dark");
    if (document.body) document.body.classList.toggle("dark", mode === "dark");
  }
  function setStyle(s) { store.set("lf_style", s); applyAppearance(); }
  function toggleMode() { store.set("lf_mode", root.getAttribute("data-lf-mode") === "dark" ? "light" : "dark"); applyAppearance(); }

  /* ------------------------------------------------------------ sidebar: collapse & width */
  function applyNav() {
    root.setAttribute("data-lf-nav", store.get("lf_nav") === "collapsed" ? "collapsed" : "open");
    var w = parseInt(store.get("lf_nav_w") || "", 10);
    if (w >= NAV_MIN && w <= NAV_MAX) root.style.setProperty("--lf-nav-w", w + "px");
    else root.style.removeProperty("--lf-nav-w");
  }
  function toggleNav() {
    store.set("lf_nav", root.getAttribute("data-lf-nav") === "collapsed" ? "open" : "collapsed");
    applyNav();
    hideTip();
  }

  applyAppearance();
  applyNav();
  if (sysDark && sysDark.addEventListener) {
    sysDark.addEventListener("change", function () { if (!store.get("lf_mode")) applyAppearance(); });
  }

  function makeResizer() {
    if (document.getElementById("lf-resizer")) return;
    var h = document.createElement("div");
    h.id = "lf-resizer";
    h.setAttribute("role", "separator");
    h.setAttribute("aria-orientation", "vertical");
    h.title = t("resize", "Drag to resize, double-click to reset");
    document.body.appendChild(h);
    var dragging = false;
    h.addEventListener("pointerdown", function (e) {
      if (root.getAttribute("data-lf-nav") === "collapsed") return;
      dragging = true;
      h.setPointerCapture(e.pointerId);
      root.classList.add("lf-resizing");
      e.preventDefault();
    });
    h.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      var inset = parseFloat(getComputedStyle(root).getPropertyValue("--lf-nav-inset")) || 0;
      var w = Math.round(Math.min(NAV_MAX, Math.max(NAV_MIN, e.clientX - inset)));
      root.style.setProperty("--lf-nav-w", w + "px");
    });
    function end(e) {
      if (!dragging) return;
      dragging = false;
      root.classList.remove("lf-resizing");
      try { h.releasePointerCapture(e.pointerId); } catch (err) {}
      var w = parseInt(getComputedStyle(root).getPropertyValue("--lf-nav-w"), 10);
      if (w) store.set("lf_nav_w", String(w));
    }
    h.addEventListener("pointerup", end);
    h.addEventListener("pointercancel", end);
    h.addEventListener("dblclick", function () {
      store.set("lf_nav_w", String(NAV_DEFAULT));
      applyNav();
    });
  }

  /* Gradio toggles body.dark from the system setting on mount; keep ours authoritative. */
  function guardBodyClass() {
    new MutationObserver(function () {
      var want = root.getAttribute("data-lf-mode") === "dark";
      if (document.body.classList.contains("dark") !== want) document.body.classList.toggle("dark", want);
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    applyAppearance();
  }

  /* ------------------------------------------------------------ i18n from the server-rendered index */
  var I18N = null;
  function loadIndex() {
    var el = document.getElementById("lf-index");
    if (!el) return;
    try { I18N = JSON.parse(el.getAttribute("data-i18n") || "{}"); } catch (e) { I18N = {}; }
  }
  function t(key, fallback) {
    if (!I18N) loadIndex();
    return (I18N && I18N[key]) || fallback || key;
  }

  /* ------------------------------------------------------------ pages */
  function go(page, keepScroll) {
    if (PAGES.indexOf(page) < 0) page = "model";
    var changed = root.getAttribute("data-lf-page") !== page;
    root.setAttribute("data-lf-page", page);
    store.set("lf_page", page);
    if (location.hash !== "#" + page) {
      try { history.replaceState(null, "", "#" + page); } catch (e) {}
    }
    if (changed && !keepScroll) window.scrollTo(0, 0);
  }
  var initial = (location.hash || "").replace("#", "") || store.get("lf_page") || "model";
  root.setAttribute("data-lf-page", PAGES.indexOf(initial) >= 0 ? initial : "model");
  window.addEventListener("hashchange", function () {
    var p = (location.hash || "").replace("#", "");
    if (PAGES.indexOf(p) >= 0) go(p);
  });

  /* ------------------------------------------------------------ parameters */
  function labelNode(block) {
    return block.querySelector('[data-testid="block-info"]')
      || block.querySelector("label > span")
      || block.querySelector(".label-text");
  }
  function paramOf(id) {
    var m = /^lfp-([a-z]+)-(.+)$/.exec(id || "");
    return m ? m[1] + "." + m[2] : null;
  }
  function pageOfParam(id) {
    var tab = (id || "").split("-")[1];
    return { top: "model", train: "train", eval: "eval", infer: "infer", export: "export" }[tab] || null;
  }
  function flash(el, cls) {
    el.classList.remove(cls);
    void el.offsetWidth;
    el.classList.add(cls);
    setTimeout(function () { el.classList.remove(cls); }, 2600);
  }
  function decorate() {
    /* long help texts are clamped to one line; keep the full text as a tooltip */
    var infos = document.querySelectorAll('[data-testid="block-info"] + div .md');
    for (var j = 0; j < infos.length; j++) {
      var txt = infos[j].textContent.trim();
      if (txt && infos[j].getAttribute("title") !== txt) infos[j].setAttribute("title", txt);
    }
  }
  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () { pending = false; decorate(); });
  }
  function revealParam(id) {
    var el = document.getElementById(id);
    if (!el) return;
    var page = pageOfParam(id);
    if (page) go(page, true);
    var acc = el.closest(".lf-acc");
    var delay = 90;
    if (acc) {
      var head = acc.querySelector(":scope > .label-wrap");
      if (head && !head.classList.contains("open")) { head.click(); delay = 260; }
    }
    setTimeout(function () {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      flash(el, "lf-spot");
      var inp = el.querySelector("textarea, input:not([type=range]):not([type=checkbox])");
      if (inp) { try { inp.focus({ preventScroll: true }); } catch (e) {} }
    }, delay);
  }

  /* ------------------------------------------------------------ tooltips */
  var tip = null;
  function showTip(el) {
    if (el.hasAttribute("data-lf-tip-nav") && root.getAttribute("data-lf-nav") !== "collapsed") return hideTip();
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "lf-tip";
      document.body.appendChild(tip);
    }
    tip.textContent = el.getAttribute("data-lf-tip");
    tip.style.display = "block";
    var r = el.getBoundingClientRect();
    if (el.hasAttribute("data-lf-tip-nav")) {
      tip.style.left = (r.right + 10) + "px";
      tip.style.top = (r.top + r.height / 2 - tip.offsetHeight / 2) + "px";
      return;
    }
    var w = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8) + "px";
    tip.style.top = (r.top - tip.offsetHeight - 8) + "px";
  }
  function hideTip() { if (tip) tip.style.display = "none"; }

  /* ------------------------------------------------------------ command palette */
  var pal = null, palItems = [], palSel = 0, palSource = [];

  function norm(s) { return (s || "").toLowerCase().replace(/\s+/g, " ").trim(); }
  function score(hay, q) {
    if (!q) return 1;
    hay = norm(hay);
    var i = hay.indexOf(q);
    if (i === 0) return 100;
    if (i > 0) return 80 - Math.min(i, 40);
    var terms = q.split(" "), all = true, s = 0;
    for (var k = 0; k < terms.length; k++) {
      if (!terms[k]) continue;
      var p = hay.indexOf(terms[k]);
      if (p < 0) { all = false; break; }
      s += 20 - Math.min(p, 15);
    }
    if (all && terms.length > 1) return s;
    var j = 0;
    for (var c = 0; c < hay.length && j < q.length; c++) if (hay[c] === q[j]) j++;
    return j === q.length && q.length > 1 ? 8 : 0;
  }

  function collect() {
    loadIndex();
    var items = [];
    var pages = (I18N && I18N.pages) || {};
    PAGES.forEach(function (p) {
      if (!document.getElementById("lf-page-" + p)) return;
      items.push({ g: "p_pages", title: pages[p] || p, sub: "", key: p, run: function () { go(p); } });
    });
    var seen = {};
    document.querySelectorAll('[id^="lfp-"]').forEach(function (el) {
      var page = pageOfParam(el.id);
      if (!page || seen[el.id]) return;
      var lab = labelNode(el);
      if (!lab || el.classList.contains("lf-acc")) return;
      var text = (lab.textContent || "").trim();
      if (!text) return;
      seen[el.id] = 1;
      var group = "";
      var acc = el.closest(".lf-acc");
      if (acc) {
        var h = acc.querySelector(":scope > .label-wrap > span");
        group = h ? h.firstChild && h.firstChild.textContent : "";
      } else {
        var card = el.closest(".lf-card");
        var ct = card && card.querySelector(".lf-card-h .t");
        group = ct ? ct.firstChild.textContent : "";
      }
      items.push({ g: "p_params", title: text, sub: (pages[page] || page) + (group ? " · " + group : ""),
        key: paramOf(el.id), run: function () { revealParam(el.id); } });
    });
    items.push({ g: "p_act", title: t("a_style", "Switch style"), sub: "", key: "style",
      run: function () { setStyle(root.getAttribute("data-lf-style") === "soft" ? "eng" : "soft"); } });
    items.push({ g: "p_act", title: t("a_mode", "Toggle dark / light"), sub: "", key: "mode", run: toggleMode });
    items.push({ g: "p_act", title: t("a_nav", "Collapse sidebar"), sub: "Ctrl+B", key: "nav", run: toggleNav });
    return items;
  }

  function buildPalette() {
    pal = document.createElement("div");
    pal.className = "lf-pal-back";
    pal.innerHTML =
      '<div class="lf-pal" role="dialog" aria-modal="true">' +
      '<div class="lf-pal-in"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="1.8" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>' +
      '<input type="text" spellcheck="false" autocomplete="off"/><kbd>Esc</kbd></div>' +
      '<div class="lf-pal-list" role="listbox"></div>' +
      '<div class="lf-pal-foot"><span><kbd>↑</kbd><kbd>↓</kbd> <em class="nav"></em></span>' +
      '<span><kbd>Enter</kbd> <em class="open"></em></span><span><kbd>Esc</kbd> <em class="close"></em></span></div></div>';
    document.body.appendChild(pal);
    var input = pal.querySelector("input");
    input.addEventListener("input", function () { palSel = 0; renderPalette(); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { palSel = Math.min(palSel + 1, palItems.length - 1); highlight(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { palSel = Math.max(palSel - 1, 0); highlight(); e.preventDefault(); }
      else if (e.key === "Enter") { choose(palSel); e.preventDefault(); }
      else if (e.key === "Escape") { closePalette(); e.preventDefault(); }
    });
    pal.addEventListener("mousedown", function (e) { if (e.target === pal) closePalette(); });
    pal.querySelector(".lf-pal-list").addEventListener("click", function (e) {
      var it = e.target.closest(".lf-pal-it");
      if (it) choose(parseInt(it.getAttribute("data-i"), 10));
    });
    pal.querySelector(".lf-pal-list").addEventListener("mousemove", function (e) {
      var it = e.target.closest(".lf-pal-it");
      if (it) { var i = parseInt(it.getAttribute("data-i"), 10); if (i !== palSel) { palSel = i; highlight(); } }
    });
  }

  function openPalette() {
    if (!pal) buildPalette();
    loadIndex();
    palSource = collect();
    var input = pal.querySelector("input");
    input.placeholder = t("p_ph", "Search…");
    pal.querySelector(".lf-pal-foot .nav").textContent = t("p_nav", "navigate");
    pal.querySelector(".lf-pal-foot .open").textContent = t("p_open", "open");
    pal.querySelector(".lf-pal-foot .close").textContent = t("p_close", "close");
    input.value = "";
    palSel = 0;
    pal.classList.add("open");
    renderPalette();
    setTimeout(function () { input.focus(); }, 10);
  }
  function closePalette() { if (pal) pal.classList.remove("open"); }

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function renderPalette() {
    var q = norm(pal.querySelector("input").value);
    var scored = [];
    palSource.forEach(function (it) {
      var s = Math.max(score(it.title, q), score(it.key, q) * 0.9, score(it.sub, q) * 0.5);
      if (!q && it.g === "p_params") s = 0;  // empty query: pages and actions only
      if (s > 0) scored.push({ it: it, s: s });
    });
    var order = { p_pages: 0, p_params: 1, p_act: 2 };
    scored.sort(function (a, b) { return q ? (b.s - a.s) || (order[a.it.g] - order[b.it.g]) : order[a.it.g] - order[b.it.g]; });
    var groups = {}, flat = [];
    scored.slice(0, 60).forEach(function (x) { (groups[x.it.g] = groups[x.it.g] || []).push(x.it); });
    var html = "";
    ["p_pages", "p_params", "p_act"].forEach(function (g) {
      if (!groups[g]) return;
      html += '<div class="lf-pal-cap">' + esc(t(g, g)) + "</div>";
      groups[g].forEach(function (it) {
        html += '<div class="lf-pal-it" role="option" data-i="' + flat.length + '"><span class="tt">' + esc(it.title) +
          "</span>" + (it.sub ? '<span class="sb">' + esc(it.sub) + "</span>" : "") +
          (g === "p_params" ? "<code>" + esc(it.key) + "</code>" : "") + "</div>";
        flat.push(it);
      });
    });
    if (!flat.length) html = '<div class="lf-pal-empty">' + esc(t("p_empty", "No matches")) + "</div>";
    palItems = flat;
    pal.querySelector(".lf-pal-list").innerHTML = html;
    highlight();
  }
  function highlight() {
    var list = pal.querySelector(".lf-pal-list");
    list.querySelectorAll(".lf-pal-it").forEach(function (el) {
      var on = parseInt(el.getAttribute("data-i"), 10) === palSel;
      el.classList.toggle("sel", on);
      if (on) el.scrollIntoView({ block: "nearest" });
    });
  }
  function choose(i) {
    var it = palItems[i];
    if (!it) return;
    closePalette();
    setTimeout(it.run, 20);
  }

  /* ------------------------------------------------------------ events */
  document.addEventListener("click", function (e) {
    var el = e.target;
    if (!(el instanceof Element)) return;
    if (el.closest("[data-lf-palette]")) { e.preventDefault(); openPalette(); return; }
    if (el.closest("[data-lf-nav-toggle]")) { e.preventDefault(); toggleNav(); return; }
    var nav = el.closest("[data-lf-go]");
    if (nav) { e.preventDefault(); go(nav.getAttribute("data-lf-go")); return; }
    var st = el.closest("[data-lf-style-set]");
    if (st) { setStyle(st.getAttribute("data-lf-style-set")); return; }
    if (el.closest("[data-lf-style-cycle]")) { setStyle(root.getAttribute("data-lf-style") === "soft" ? "eng" : "soft"); return; }
    if (el.closest("[data-lf-mode-toggle]")) toggleMode();
  }, true);

  document.addEventListener("keydown", function (e) {
    var mod = e.ctrlKey || e.metaKey;
    if (mod && (e.key === "k" || e.key === "K")) { e.preventDefault(); if (pal && pal.classList.contains("open")) closePalette(); else openPalette(); return; }
    if (mod && (e.key === "b" || e.key === "B")) { e.preventDefault(); toggleNav(); }
  });

  document.addEventListener("mouseover", function (e) {
    var el = e.target instanceof Element ? e.target.closest("[data-lf-tip]") : null;
    if (el) showTip(el); else hideTip();
  });
  document.addEventListener("focusin", function (e) {
    var el = e.target instanceof Element ? e.target.closest("[data-lf-tip]") : null;
    if (el) showTip(el); else hideTip();
  });
  window.addEventListener("scroll", hideTip, { passive: true });

  function boot() {
    guardBodyClass();
    makeResizer();
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
    schedule();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.LF = { go: go, reveal: revealParam, palette: openPalette, toggleNav: toggleNav, setStyle: setStyle,
    toggleMode: toggleMode };
})();
