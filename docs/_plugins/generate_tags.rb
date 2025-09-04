# _plugins/generate_tags.rb
# Builds /tags-<slug>/index.html for:
# - every tag found in posts
# - every label listed in _data/tag_groups.yml
#
# The layout used is _layouts/tag.html, which itself extends your default layout.

module Jekyll
  class TagPage < Page
    def initialize(site, base, dir, tag, has_posts)
      @site = site
      @base = base
      @dir  = dir  # e.g., "tags-indexing"
      @name = "index.html"

      self.process(@name)
      self.read_yaml(File.join(base, "_layouts"), "tag.html")

      # Pass data to the layout
      self.data["title"]     = "Posts tagged: #{tag}"
      self.data["tag"]       = tag
      self.data["tag_slug"]  = Jekyll::Utils.slugify(tag)
      self.data["has_posts"] = has_posts
    end
  end

  class TagGenerator < Generator
    safe true
    priority :low

    def generate(site)
      # Ensure the tag layout exists
      return unless site.layouts.key?("tag")

      # Tags discovered from posts
      discovered = site.tags.keys

      # Tags declared in _data/tag_groups.yml (if any)
      declared = []
      if site.data.key?("tag_groups")
        groups = site.data["tag_groups"]
        if groups.is_a?(Array)
          groups.each do |g|
            items = g["items"]
            declared.concat(items) if items.is_a?(Array)
          end
        end
      end

      # Union (preserve human-readable casing for display)
      all_labels = (discovered + declared).uniq

      all_labels.each do |label|
        slug      = Jekyll::Utils.slugify(label)
        has_posts = site.tags.key?(label) && site.tags[label].is_a?(Array) && !site.tags[label].empty?
        dir       = "tags-#{slug}" # final URL under /blog: /blog/tags-<slug>/

        site.pages << TagPage.new(site, site.source, dir, label, has_posts)
      end
    end
  end
end