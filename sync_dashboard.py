"""SI disclosure: written with assistance from an AI coding assistant (a Super Intelligence tool);
directed, reviewed and tested by the HealthLoop team. See README "Use of Super Intelligence".

Copy the dashboard (healthloop.html, the published artifact source) into the package,
wrapping it in a full HTML document so the Python server can serve it directly.
Usage: python sync_dashboard.py path/to/healthloop.html"""
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
body = src.replace("</style>\n", "</style>\n</head>\n<body>\n", 1) + "\n</body>\n</html>\n"
out = Path(__file__).parent / "healthloop" / "static" / "index.html"
out.write_text(head + body)
print("wrote", out)
