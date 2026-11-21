// Desktop bridge: route fetch("/api/...") to pywebview.api.* (no HTTP server).
(function () {
  function makeResponse(data, ok, status) {
    const text = typeof data === "string" ? data : JSON.stringify(data);
    return {
      ok: ok !== false,
      status: status || 200,
      json: async () => (typeof data === "string" ? JSON.parse(data) : data),
      text: async () => text,
    };
  }

  window.__desktopBridgeLoaded = true;

  function call(name, args) {
    return new Promise((resolve) => {
      const deadline = Date.now() + 20000;
      function attempt() {
        if (!window.pywebview || !window.pywebview.api || !window.pywebview.api[name]) {
          if (Date.now() > deadline) {
            resolve(makeResponse({ error: "pywebview api not ready: " + name }, false, 500));
            return;
          }
          setTimeout(attempt, 50);
          return;
        }
        const fn = window.pywebview.api[name];
        if (!fn) {
          resolve(makeResponse({ error: "desktop bridge missing: " + name }, false, 500));
          return;
        }
        Promise.resolve(fn.apply(null, args || []))
          .then((data) => resolve(makeResponse(data, true, 200)))
          .catch((err) => {
            const msg = String((err && err.message) || err || "bridge error");
            resolve(makeResponse({ error: msg }, false, 500));
          });
      }
      attempt();
    });
  }

  function bodyJson(init) {
    try {
      return JSON.parse((init && init.body) || "{}");
    } catch (err) {
      return {};
    }
  }

  function queryParam(u, key) {
    try {
      return new URL(u, window.location.href).searchParams.get(key) || "";
    } catch (err) {
      return "";
    }
  }

  function patchFetch() {
    const origFetch = window.fetch;
    window.fetch = function (url, init) {
      const u = String(url);
      if (u.startsWith("/api/")) {
        if (u.startsWith("/api/defaults")) {
          return call("api_defaults", []);
        }
        if (u.startsWith("/api/pick-folder")) {
          return call("api_pick_folder", [String(bodyJson(init).initial || "")]);
        }
        if (u.startsWith("/api/build-command")) {
          return call("api_build_command", [bodyJson(init)]);
        }
        if (u.startsWith("/api/perturbation-inputs")) {
          return call("api_perturbation_inputs", [bodyJson(init)]);
        }
        if (u.startsWith("/api/run")) {
          return call("api_run", [bodyJson(init)]);
        }
        if (u.startsWith("/api/resume")) {
          return call("api_resume", [bodyJson(init)]);
        }
        if (u.startsWith("/api/stop")) {
          return call("api_stop", [bodyJson(init)]);
        }
        if (u.startsWith("/api/jobs/")) {
          return call("api_job", [u.split("/").pop().split("?")[0]]);
        }
        if (u.startsWith("/api/jobs")) {
          return call("api_jobs", []);
        }
        if (u.startsWith("/api/preview")) {
          return call("api_preview", [queryParam(u, "path")]);
        }
        if (u.startsWith("/api/browse")) {
          return call("api_browse", [queryParam(u, "path")]);
        }
      }
      return origFetch(url, init);
    };
    window.__desktopBridgePatched = true;
  }

  // Install the fetch override immediately; individual calls wait for the
  // pywebview JS API, which pywebview injects only after page load.
  patchFetch();
})();
