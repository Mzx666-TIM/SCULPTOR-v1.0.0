# Vendored packages

## proxy_tools

`proxy_tools/__init__.py` is the unmodified `proxy_tools` 0.1.0 module
(copyright Armin Ronacher / Jonathan Tushman, BSD licence, extracted from
Werkzeug).  SHA256:

    0VOdleGnE8BoyoHULgR7LHZWiWTPJ3WW1OGe-yL0dr4   (from the wheel RECORD)
    d1539d95e1a713c068ca81d42e047b2c76568964cf277596d4e19efb22f476be  (file)

It is shipped inside the package because `pywebview` imports it at start-up
and at least one package mirror serves a **broken** `proxy_tools` wheel: the
archive contains only `proxy_tools-0.1.0.dist-info/` and no module file, so
`pip install pywebview` reports success while `import webview` fails with
`ModuleNotFoundError: No module named 'proxy_tools'`.  Because the
distribution metadata is present, pip considers the requirement satisfied and
re-running the install does not repair it.

`setup_env.ps1` checks the import and, when it fails, copies this directory into
the environment's `site-packages`, so the native control-panel window works
even on a machine whose mirror cannot deliver the package.
