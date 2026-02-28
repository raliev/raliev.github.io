rm -rf ../blog
git rm -r --cached ../blog
bundle exec jekyll build && mv blog ../ &&  git add ../blog/ && echo "SUCCESS!"
