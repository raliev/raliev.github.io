rm -rf ../blog
git rm -r --cached ../blog
jekyll build && mv blog ../ && mkdir ../blog/pdfs && git add ../blog/ && find ../tex -name "*.pdf" | while read p; do cp $p ../blog/pdfs/; done && echo "SUCCESS!"
