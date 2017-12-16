(defproject dragan-rocks "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [uncomplicate/clojurecl "0.7.1"]
                 [uncomplicate/neanderthal "0.17.0"
                  :exclusions [org.clojure/tools.reader
                               org.clojure/tools.analyzer.jvm
                               uncomplicate/commons
                               org.clojure/core.async]]
                 [net.mikera/core.matrix "0.61.0"]
                 [uncomplicate/fluokitten "0.6.0"]
                 [org.clojure/core.async "0.3.465"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [denisovan "0.1.0-SNAPSHOT"]])