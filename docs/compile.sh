rm -rf ../blog
git rm -r --cached ../blog
jekyll build && mv blog ../ && git add ../blog/ && echo "SUCCESS!"
