(ns dragan-rocks.cuda
  (:require [uncomplicate.clojurecuda.core :as core]
            [uncomplicate.clojurecuda.info :as info]
            [uncomplicate.clojurecuda.nvrtc :as nv]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [asum copy dot]]
            [uncomplicate.neanderthal.cuda :as cuda]
            [uncomplicate.neanderthal.native :refer [dv]]))

(core/init)

(map info/info (map core/device (range (core/device-count))))

(cuda/with-default-engine
  (with-release [gpu-x (cuda/cuv 1 -2 5)]
    (asum gpu-x)))

(time
 (cuda/with-default-engine
   (with-release [gpu-x (cuda/cuv (range 100000000))
                  gpu-y (copy gpu-x)]
     (dot gpu-x gpu-y))))

(time
 (let [x (dv (range 100000000))
       y (copy x)]
   (dot x y)))
