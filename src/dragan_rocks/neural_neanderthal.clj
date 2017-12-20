(ns dragan-rocks.neural-neanderthal
  (:require [uncomplicate.neanderthal
             [native :refer [dv dge fge dtr native-float fv]]
             [core :refer [mm asum dot axpy! mm! transfer! copy] :as n-core]
             [linalg :as lin]
             [opencl :refer [with-default-engine clv clge] :as opencl]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [with-default finish! devices
                           platforms with-platform
                           sort-by-cl-version with-context
                           context with-queue] :as clojurecl]
             [info :as info]
             [legacy :refer [with-default-1 command-queue-1] :as legacy]]))

(map info/info (platforms))

(map info/info (devices (first (platforms))))

(devices (first (platforms)))

(def input-data
  (dge 4 3 [0 0 1
            0 1 1
            1 0 1
            1 1 1] {:layout :row}))

(def output-data
  (dge 4 1 [0 1 1 0]))

(defn random-matrix [nrows ncol]
  (dge nrows ncol (repeatedly #(Math/random))))

(mm (random-matrix 3 4) (random-matrix 4 1))

(let [cs (dge 3 1)]
  (lin/sv! (dge 3 3 [1 2 0
                     0 1 -1
                     1 1 2])
           cs)
  cs)

(let [a (dge 4 4 [1 0 0 0
                  2 0 0 0
                  0 1 0 0
                  0 0 1 0] )]
  (lin/svd a))

(dot (dv [2 3]) (dv [1 2]));; => 8.0

(with-default-1
  (with-default-engine
    (asum (fv 1 -2 3))))
;; => 0.0

(with-default-1
  (with-default-engine
    (with-release [gpu-x (clv 1 -2 5)]
      (asum gpu-x))))
;; => 8.0

(with-default-1
  (with-default-engine
    (with-release [gpu-x (transfer! (fv -12 3) (clv 2))]
      (asum gpu-x))))
;; => 15.0

(time
 (with-default-1
  (with-default-engine
    (with-release [gpu-x (clv (range 100000000))
                   gpu-y (copy gpu-x)]
      (dot gpu-x gpu-y)))))

; Native method as comparison
(time
 (let [x (dv (range 100000000))
       y (copy x)]
  (dot x y)))

(with-platform (first (platforms))
  (let [dev (first (sort-by-cl-version (devices :gpu)))]
    (with-context (context [dev])
      (with-queue (command-queue-1 dev)
        (with-default-engine
          (with-release [gpu-x (clv 1 -2 5)]
            (asum gpu-x)))))))
