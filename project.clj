(defproject dragan-rocks "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[uncomplicate/clojurecl "0.7.1"]
                 [uncomplicate/neanderthal "0.17.1"]
                 [net.mikera/core.matrix "0.61.0"]
                 [uncomplicate/fluokitten "0.6.0"]
                 [org.clojure/clojure "1.9.0-RC1"]
                 [org.clojure/core.async "0.3.465"]]
  :profiles {:dev {:global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[net.mikera/core.matrix "0.61.0" :classifier "tests"]
                                  [net.mikera/vectorz-clj "0.43.1" :exclusions [net.mikera/core.matrix]]
                                  [criterium/criterium "0.4.3"]]}})
