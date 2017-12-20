(defproject dragan-rocks "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [uncomplicate/clojurecl "0.7.2"]
                 [uncomplicate/neanderthal "0.17.2"
                  :exclusions [org.clojure/tools.reader
                               org.clojure/tools.analyzer.jvm
                               uncomplicate/commons
                               org.clojure/core.async]]
                 [net.mikera/core.matrix "0.61.0"]
                 [org.clojure/core.async "0.3.465"]
                 [denisovan "0.1.0-SNAPSHOT"]
                 [criterium "0.4.4"]
                 [midje "1.9.0"]])
