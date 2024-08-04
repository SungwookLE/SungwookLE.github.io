source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-sitemap"
  gem "jemoji"
  gem "jekyll-toc"
  # jekyll-admin은 GitHub Pages에서 지원되지 않으므로 주석 처리합니다
  # gem 'jekyll-admin'
end

# Windows와 JRuby는 zoneinfo 파일을 포함하지 않으므로 tzinfo-data gem과 관련 라이브러리를 번들링합니다
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Windows에서 디렉토리 감시 성능 향상을 위한 gem
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]