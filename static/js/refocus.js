// ReFoCUS project page — interactivity
(function () {
  "use strict";

  // ---------- Copy BibTeX ----------
  function setupCopy() {
    var btn = document.getElementById("copy-bibtex");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var pre = document.getElementById("bibtex-code");
      var text = pre ? pre.innerText : "";
      var done = function () {
        btn.classList.add("copied");
        var label = btn.querySelector(".copy-label");
        var old = label ? label.textContent : "";
        if (label) label.textContent = "Copied!";
        setTimeout(function () {
          btn.classList.remove("copied");
          if (label) label.textContent = old;
        }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {
          fallbackCopy(text); done();
        });
      } else {
        fallbackCopy(text); done();
      }
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
  }

  // ---------- Scroll to top ----------
  function setupScrollTop() {
    var btn = document.getElementById("scroll-top");
    if (!btn) return;
    window.addEventListener("scroll", function () {
      if (window.scrollY > 600) btn.classList.add("show");
      else btn.classList.remove("show");
    }, { passive: true });
    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // ---------- Qualitative tabs (with timed auto-advance) ----------
  function setupQualTabs() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll(".qual-tab"));
    var panels = document.querySelectorAll(".qual-panel");
    if (!tabs.length) return;

    var INTERVAL = 7000;        // ~7s per example
    var timer = null;
    var paused = false;         // true while hovered / focused
    // Respect reduced-motion: start paused (no auto-advance unless the user interacts).
    var reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function activate(tab) {
      var id = tab.getAttribute("data-target");
      tabs.forEach(function (t) { t.classList.remove("active"); });
      panels.forEach(function (p) { p.classList.remove("active"); });
      tab.classList.add("active");
      var panel = document.getElementById(id);
      if (panel) panel.classList.add("active");
    }

    function currentIndex() {
      for (var i = 0; i < tabs.length; i++) {
        if (tabs[i].classList.contains("active")) return i;
      }
      return 0;
    }

    function advance() {
      activate(tabs[(currentIndex() + 1) % tabs.length]); // wrap (last -> first)
    }

    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    function start() {
      stop();
      if (reduceMotion || paused || tabs.length < 2) return;
      timer = setInterval(advance, INTERVAL);
    }
    function restart() { start(); } // reset the countdown from now

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        activate(tab);
        restart(); // manual pick resets the timer — no immediate jump afterwards
      });
    });

    // Pause while the pointer is over the qual section or a tab has keyboard focus.
    var scope = tabs[0].closest("section") || tabs[0].parentNode;
    if (scope) {
      scope.addEventListener("mouseenter", function () { paused = true; stop(); });
      scope.addEventListener("mouseleave", function () { paused = false; start(); });
      scope.addEventListener("focusin", function () { paused = true; stop(); });
      scope.addEventListener("focusout", function () { paused = false; start(); });
    }

    start();
  }

  // ---------- Count-up stats ----------
  function setupCounters() {
    var nums = document.querySelectorAll(".stat-num[data-target]");
    if (!nums.length || !("IntersectionObserver" in window)) {
      nums.forEach(function (n) { n.textContent = n.getAttribute("data-display") || n.getAttribute("data-target"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var el = entry.target;
        io.unobserve(el);
        var target = parseFloat(el.getAttribute("data-target"));
        var prefix = el.getAttribute("data-prefix") || "";
        var suffix = el.getAttribute("data-suffix") || "";
        var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
        var dur = 1100, start = null;
        function step(ts) {
          if (start === null) start = ts;
          var p = Math.min((ts - start) / dur, 1);
          var eased = 1 - Math.pow(1 - p, 3);
          var val = (target * eased).toFixed(decimals);
          el.textContent = prefix + val + suffix;
          if (p < 1) requestAnimationFrame(step);
          else el.textContent = prefix + target.toFixed(decimals) + suffix;
        }
        requestAnimationFrame(step);
      });
    }, { threshold: 0.4 });
    nums.forEach(function (n) { io.observe(n); });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupCopy();
    setupScrollTop();
    setupQualTabs();
    setupCounters();
  });
})();
