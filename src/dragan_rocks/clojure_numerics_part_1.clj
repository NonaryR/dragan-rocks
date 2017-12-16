(ns dragan-rocks.clojure-numerics-part-1
  (:require [uncomplicate.neanderthal
             [native :refer [dv dge fge dtr native-float]]
             [opencl :as ocl]
             [core :refer [copy copy! row submatrix scal! transfer! transfer mrows ncols nrm2 mm cols view-tr xpy scal axpy! zero]]
             [real :refer [entry entry!]]
             [linalg :refer [trf tri det]]]
           [uncomplicate.commons.core :refer [with-release]]
           [uncomplicate.clojurecl.core :as clojurecl]
           [uncomplicate.clojurecl.info :as info]))

;; (map info/info (clojurecl/platforms))

;; (map info/info (clojurecl/devices (first (clojurecl/platforms))))


(def v1 (dv -1 2 5.2 0))
(def v2 (dv (range 22)))
(def v3 (dv -2 -3 1 0))

;; сумма v1 v3
(xpy v1 v3)
;; => #RealBlockVector[double, n:4, offset: 0, stride:1]
[  -3.00   -1.00    6.20    0.00 ]


;; скалярное произведение
(scal 2.5 v1)
;; => #RealBlockVector[double, n:4, offset: 0, stride:1]
[  -2.50    5.00   13.00    0.00 ]


;; скалярное первого вектора и сумма со вторым
(axpy! 2.5 v1 v3)
;; => #RealBlockVector[double, n:4, offset: 0, stride:1]
[  -7.00    7.00   27.00    0.00 ]


;; 7 нулей
(dv 7)
;; => #RealBlockVector[double, n:7, offset: 0, stride:1]
[   0.00    0.00    0.00    ⋯      0.00    0.00 ]

;; zeros-like
(zero v2)

(dge 3 2 [1 2 3 4 5 6] {:order :row})
;; => #RealGEMatrix[double, mxn:3x2, layout:column, offset:0]
   ▥       ↓       ↓       ┓    
   →       1.00    4.00         
   →       2.00    5.00         
   →       3.00    6.00         
   ┗                       ┛    


(row (dge 2 3 (range 6)) 1)
;; => #RealBlockVector[double, n:3, offset: 1, stride:2]
[   1.00    3.00    5.00 ]


(let [a (dge 2 3 (range 6))
      b (submatrix a 0 1 1 2)]
  (scal! 100 b)
  a)
;; => #RealGEMatrix[double, mxn:2x3, layout:column, offset:0]
   ▥       ↓       ↓       ↓       ┓    
   →       0.00  200.00  400.00         
   →       1.00    3.00    5.00         
   ┗                               ┛    


(let [a (dge 3 2 (range 6))
      b (dge 3 2)]
  (copy! a b)
  (entry! a 1 1 800)
  (copy b))
;; => #RealGEMatrix[double, mxn:3x2, layout:column, offset:0]
   ▥       ↓       ↓       ┓    
   →       0.00    3.00         
   →       1.00    4.00         
   →       2.00    5.00         
   ┗                       ┛    

