(ns dragan-rocks.opencl
  (:require [midje.sweet :refer [facts => truthy]]
            [criterium.core :refer [quick-bench with-progress-reporting]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [finish!]]
             [legacy :refer [with-default-1]]]
            [uncomplicate.neanderthal
             [core :refer [asum dot axpy! mv! mm! transfer! copy]]
             [native :refer [fv fge]]
             [opencl :refer [with-default-engine clv clge]]]))

(with-default-1
  (with-default-engine
    (facts "We'll write our GPU code here, but for now here is only the plain
CPU stuff you recognize from the plain Neanderthal tutorial."

      (asum (fv 1 -2 3)) => 6.0)))

(with-default-1
  (with-default-engine
    (with-release [gpu-x (transfer! (fv 1 -2 3) (clv 3))]
      (facts
        "Create a vector on the device, write into it the data from the host vector
and compute the sum of absolute values."
        (asum gpu-x))) => 6.0))

(with-default-1
  (with-default-engine
    (with-release [host-x (fv 1 -2 3)
                   gpu-x (clv 1 -2 3)]
      (facts
        "Compare the speed of computing small vectors on CPU and GPU"
        (asum host-x) => 6.0
        (println "CPU:")
        (with-progress-reporting (quick-bench (asum host-x)))
        (asum gpu-x) => 6.0
        (println "GPU:")
        (with-progress-reporting (quick-bench (do (asum gpu-x) (finish!))))))))


(with-default-1
  (with-default-engine
    (let [cnt (long (Math/pow 2 20))]
      (with-release [host-x (fv (range cnt))
                     gpu-x (transfer! host-x (clv cnt))]
        (facts
          "Let's try with 2^20. That's more than a million."

          (asum host-x) => (float 5.49755126E11)
          (println "CPU:")
          (with-progress-reporting (quick-bench (asum host-x)))

          (asum gpu-x) => 5.497552896E11
          (println "GPU:")
          (with-progress-reporting (quick-bench (do (asum gpu-x) (finish!)))))))))


(with-default-1
  (with-default-engine
    ;; I had to change it to 2^28 because a recent update for my GPU driver caused
    ;; it to complain about insufficient memory, but this is probably a temporary issue.

    (let [cnt (long (Math/pow 2 28))]
      (with-release [host-x (fv (range cnt))
                     gpu-x (transfer! host-x (clv cnt))]
        (facts
         "Let's try with 2^28. That's 1GB, half the maximum that Java buffers can
currently handle. Java 9 would hopefully increase that."

         ;; note the less precise result in the CPU vector. That's because single
         ;; precision floats are not precise enough for so many accumulations.
         ;; In real life, sometimes you must use doubles in such cases.
          (asum host-x) => (float 3.6064003E16)
         (println "CPU:")
         (with-progress-reporting (quick-bench (asum host-x)))

         ;; GPU engine uses doubles for this accumulation, so the result is more precise.
          (asum gpu-x) => (float 3.6028797018963968E16)
         (println "GPU:")
         (with-progress-reporting (quick-bench (do (asum gpu-x) (finish!)))))))))


(with-default-1
  (with-default-engine
    (let [cnt (long (Math/pow 2 28))]
      (with-release [host-x (fv (range cnt))
                     host-y (copy host-x)
                     gpu-x (transfer! host-x (clv cnt))
                     gpu-y (copy gpu-x)]
        (facts
          "Let's try with a more parallel linear operation: adding two vectors.
I'll set them to 1GB each because my GPU does not have enough memory to
hold 4GB of data (it has 4GB total memory)."

          (axpy! 3 host-x host-y) => host-y
          (println "CPU:")
          (with-progress-reporting (quick-bench (axpy! 3 host-x host-y)))

          (axpy! 3 gpu-x gpu-y) => gpu-y
          (println "GPU:")
          (with-progress-reporting (quick-bench (do (axpy! 3 gpu-x gpu-y) (finish!)))))))))


(with-default-1
  (with-default-engine
    (let [cnt 8192]
      (with-release [host-a (fge cnt cnt (range (* cnt cnt)))
                     host-x (fv (range cnt))
                     host-y (copy host-x)
                     gpu-a (transfer! host-a (clge cnt cnt))
                     gpu-x (transfer! host-x (clv cnt))
                     gpu-y (copy gpu-x)]
        (facts
          "Matrix-vector multiplication. Matrices of 8192x8192 (268 MB) are usually
demanding enough."

          (mv! 3 host-a host-x 2 host-y) => host-y
          (println "CPU:")
          (with-progress-reporting (quick-bench (mv! 3 host-a host-x 2 host-y)))

          (mv! 3 gpu-a gpu-x 2 gpu-y) => gpu-y
          (println "GPU:")
          (with-progress-reporting (quick-bench (do (mv! 3 gpu-a gpu-x 2 gpu-y) (finish!)))))))))


(with-default-1
  (with-default-engine
    (let [cnt 8192]
      (with-release [host-a (fge cnt cnt (range (* cnt cnt)))
                     host-b (copy host-a)
                     host-c (copy host-a)
                     gpu-a (transfer! host-a (clge cnt cnt))
                     gpu-b (copy gpu-a)
                     gpu-c (copy gpu-a)]
        (facts
          "Matrix-matrix multiplication. Matrices of 8192x8192 (268 MB) are usually
demanding enough."

          (println "CPU:")
          (time (mm! 3 host-a host-b 2 host-c)) => host-c
          (mm! 3 gpu-a gpu-b 2 gpu-c) => gpu-c
          (finish!)
          (println "GPU:")
          (time (do (mm! 3 gpu-a gpu-b 2 gpu-c) (finish!))))))))
