(ns dragan-rocks.neural-neanderthal
  (:require [uncomplicate.neanderthal
             [native :refer [dv dge fge dtr native-float fv]]
             [core :refer [mm asum dot axpy! mm! transfer! copy] :as n-core]
             [linalg :as lin]
             [opencl :refer [with-default-engine clv clge] :as opencl]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [with-default finish!] :as clojurecl]
             [info :as info]]
            [vertigo
             [bytes :refer [direct-buffer byte-seq]]
             [structs :refer [wrap-byte-seq int8]]]))

(map info/info (clojurecl/platforms))

(map info/info (clojurecl/devices (first (clojurecl/platforms))))

(clojurecl/devices (first (clojurecl/platforms)))

(def input-data
  (dge 4 3 [0 0 1
            0 1 1
            1 0 1
            1 1 1]))

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

(n-core/dot (dv [2 3]) (dv [1 2]));; => 8.0


#_(with-default
  (with-default-engine
    (asum (fv 1 -2 3))))


#_(with-default
  (with-default-engine
    (with-release [gpu-x (transfer! (fv 1 -2 3) (clv 3))]
        (asum gpu-x))))

(def dev (first (clojurecl/devices (first (clojurecl/platforms)))))

#_(let [work-sizes (clojurecl/work-size [1])
      host-msg (direct-buffer 16)
      program-source
      "__kernel void hello_kernel(__global char16 *msg) {\n    *msg = (char16)('H', 'e', 'l', 'l', 'o', ' ',
   'k', 'e', 'r', 'n', 'e', 'l', '!', '!', '!', '\\0');\n}\n"
      ]
  (with-release
    [
     ctx (clojurecl/context [dev])
     cqueue (clojurecl/command-queue ctx dev)
     cl-msg (clojurecl/cl-buffer ctx 16 :write-only)
     prog (clojurecl/build-program! (clojurecl/program-with-source ctx [program-source]))
     hello-kernel (clojurecl/kernel prog "hello_kernel")
     ]
    (clojurecl/set-args! hello-kernel cl-msg)
    (clojurecl/enq-nd! cqueue  hello-kernel work-sizes)
    (clojurecl/enq-read! cqueue cl-msg host-msg)
    (apply str (map char
                    (wrap-byte-seq int8 (byte-seq host-msg))))))

