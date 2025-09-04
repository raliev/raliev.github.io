rm -rf ../blog
git rm -r --cached ../blog
jekyll build && mv blog ../ && mkdir ../pdfs && git add ../blog/ && find ../tex -name "*.pdf" | while read p; do cp $p ../pdfs/; done && echo "SUCCESS!"
