"""Une las partes del apunte en un solo HTML y lo convierte a PDF con Edge (o Chrome) sin interfaz.

Uso, desde la raíz del repo:
    .entorno\\Scripts\\python.exe apunte\\construir.py
"""
import re
import subprocess
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PARTES = sorted((AQUI / "partes").glob("*.html"))
HTML = AQUI / "apunte_mineria_de_datos.html"
PDF = AQUI / "Apunte_Mineria_de_Datos.pdf"

html = "".join(p.read_text(encoding="utf-8") for p in PARTES)

# Grilla de respuestas: 20 filas (Sim. 1: 1-20; Sim. 2: 1-14 y 15-28; Sim. 3: 29-43 y 44-57)
filas = ["<tr><th>Nº</th><th>Sim. 1</th><th>Nº</th><th>Sim. 2</th><th>Nº</th><th>Sim. 2</th>"
         "<th>Nº</th><th>Sim. 3</th><th>Nº</th><th>Sim. 3</th></tr>"]
for i in range(20):
    celdas = [str(i + 1)]
    celdas.append(str(i + 1) if i < 14 else "")
    celdas.append(str(i + 15) if i < 14 else "")
    celdas.append(str(i + 29) if i < 15 else "")
    celdas.append(str(i + 44) if i < 14 else "")
    fila = "".join(f"<td>{c}</td><td></td>" if c else "<td></td><td></td>" for c in celdas)
    filas.append(f"<tr>{fila}</tr>")
html = re.sub(r'<table class="grilla">.*?</table>', '<table class="grilla">' + "".join(filas) + "</table>",
              html, flags=re.S)

HTML.write_text(html, encoding="utf-8")

navegadores = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]
nav = next(n for n in navegadores if n.exists())
antes = PDF.stat().st_mtime if PDF.exists() else 0
subprocess.run([str(nav), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, timeout=180)
# Si el PDF está abierto en un visor, Windows lo bloquea y el navegador no lo reemplaza sin dar error.
if not PDF.exists() or PDF.stat().st_mtime == antes:
    raise SystemExit(f"No se pudo escribir {PDF.name}: cerrá el PDF si está abierto y volvé a ejecutar.")
print("PDF:", PDF, PDF.stat().st_size // 1024, "KB")
