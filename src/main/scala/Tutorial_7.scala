object Tutorial_7 extends App {
  /* Shape modelling with Gaussian processes and kernels
  Fabio Fehr
  21/05/2020
  * */
  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.io.{StatisticalModelIO, MeshIO}
  import scalismo.statisticalmodel._
  import scalismo.numerics.UniformMeshSampler3D
  import scalismo.kernels._
  import breeze.linalg.{DenseMatrix, DenseVector}

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()


  val referenceMesh = MeshIO.readMesh(new java.io.File("datasets/lowResPaola.ply")).get

  val modelGroup = ui.createGroup("gp-model")
  val referenceView = ui.show(modelGroup, referenceMesh, "reference")


  // The mean:
  // As we are modelling deformation fields, the mean of the Gaussian process will, of course,
  // itself be a deformation field. In terms of shape models, we can think of the mean function
  // as the deformation field that deforms our reference mesh into the mean shape.

  //If the reference shape that we choose corresponds approximately to an average shape,
  // and we do not have any further knowledge about our shape space, it is entirely reasonable
  // to use a zero mean; I.e. a deformation field which applies to every point a zero deformation.

  val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))

  // The covariance function: (Kernel)
  // Formally, it is a symmetric, positive semi-definite function which defines the covariance between points
  // x xprime of the reference domain.

////  abstract class MatrixValuedPDKernel[_3D]() {
////
////    def outputDim: Int; // 3D vectors thus its 3
////    def domain: Domain[_3D]; // the set of points its defined
////    def k(x: Point[_3D], y: Point[_3D]): DenseMatrix[Double]; // covariance function
////  }
//
//  // common kernel
//  case class MatrixValuedGaussianKernel3D(sigma2 : Double) extends MatrixValuedPDKernel[_3D]() {
//
//    override def outputDim: Int = 3
//    override def domain: Domain[_3D] = RealSpace[_3D];
//
//    override def k(x: Point[_3D], y: Point[_3D]): DenseMatrix[Double] = {
//      DenseMatrix.eye[Double](outputDim) * Math.exp(- (x - y).norm2 / sigma2)
//    }
//  }

  // This is already implemented in scalismo
  val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 100.0)

  // now we make it matrix values
  //val matrixValuedGaussianKernel = DiagonalKernel(scalarValuedGaussianKernel, scalarValuedGaussianKernel, scalarValuedGaussianKernel)
  val matrixValuedGaussianKernel = DiagonalKernel(scalarValuedGaussianKernel, 3)

  // now we can build the GP
  val gp = GaussianProcess(zeroMean, matrixValuedGaussianKernel)

  val sampleGroup = ui.createGroup("samples")
  val sample = gp.sampleAtPoints(referenceMesh.pointSet)
  ui.show(sampleGroup, sample, "gaussianKernelGP_sample")

  val interpolatedSample = sample.interpolate(NearestNeighborInterpolator())
  val deformedMesh = referenceMesh.transform((p : Point[_3D]) => p + interpolatedSample(p))
  ui.show(sampleGroup, deformedMesh, "deformed mesh")

  // Low-rank approximation
  // enever we create a sample using the sampleAtPoints method of the Gaussian process, internally a matrix of
  // dimensionality nd x nd (n is points and d is dimensionality) - runs out of memory

  // Thus we do a low rank approx like PCA we use the approximateGPCholesky
  // Computes a finite-rank approximation of the Gaussian Process using a Pivoted Cholesky approximation.
  //  choses a rank such that a certain error is achived... MORE OF THESE THUS MORE NON-LINEARITY??
  // (The error is measures in terms of the variance of the Gaussian process, approximated on the points of the reference Mesh)

  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    referenceMesh.pointSet,
    gp,
    relativeTolerance = 0.01,
    interpolator = NearestNeighborInterpolator()
  )

  val  defField : Field[_3D, EuclideanVector[_3D]]= lowRankGP.sample
  referenceMesh.transform((p : Point[_3D]) => p + defField(p)) // This is how we take the reference then transform it!

  val ssm = StatisticalMeshModel(referenceMesh, lowRankGP)
  val ssmView = ui.show(modelGroup, ssm, "group")


  // Building more interesting kernels

  val pcaModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/lowresModel.h5")).get

  // This is the covariance matrix

  val gpSSM = pcaModel.gp.interpolate(NearestNeighborInterpolator())
  val covSSM : MatrixValuedPDKernel[_3D] = gpSSM.cov
  //  //additional variance using a Gaussian kernel and add it to the sample covariance kernel.
  val augmentedCov = covSSM + DiagonalKernel(GaussianKernel[_3D](100.0), 3)

  val augmentedGP = GaussianProcess(gpSSM.mean, augmentedCov)

  val lowRankAugmentedGP = LowRankGaussianProcess.approximateGPCholesky(
    referenceMesh.pointSet,
    augmentedGP,
    relativeTolerance = 0.01,
    interpolator = NearestNeighborInterpolator(),
  )
  // now we have a plain PCA model and then a augmented PCA model
  val augmentedSSM = StatisticalMeshModel(pcaModel.referenceMesh, lowRankAugmentedGP)

  // Changepoint kernel
  case class ChangePointKernel(kernel1 : MatrixValuedPDKernel[_3D], kernel2 : MatrixValuedPDKernel[_3D])
    extends MatrixValuedPDKernel[_3D]() {

    override def domain = RealSpace[_3D]
    val outputDim = 3

    def s(p: Point[_3D]) =  1.0 / (1.0 + math.exp(-p(0)))
    def k(x: Point[_3D], y: Point[_3D]) = {
      val sx = s(x)
      val sy = s(y)
      kernel1(x,y) * sx * sy + kernel2(x,y) * (1-sx) * (1-sy)
    }

  }
  val gk1 = DiagonalKernel(GaussianKernel[_3D](100.0), 3)
  val gk2 = DiagonalKernel(GaussianKernel[_3D](10.0), 3)
  val changePointKernel = ChangePointKernel(gk1, gk2)
  val gpCP = GaussianProcess(zeroMean, changePointKernel)
  val sampleCP =  gpCP.sampleAtPoints(referenceMesh.pointSet)
  ui.show(sampleGroup, sampleCP, "ChangePointKernelGP_sample")

  // Symmetric kernel

  case class xMirroredKernel(kernel : PDKernel[_3D]) extends PDKernel[_3D] {
    override def domain = kernel.domain
    override def k(x: Point[_3D], y: Point[_3D]) = kernel(Point(x(0) * -1.0 ,x(1), x(2)), y)

  }

  def symmetrizeKernel(kernel : PDKernel[_3D]) : MatrixValuedPDKernel[_3D] = {
    val xmirrored = xMirroredKernel(kernel)
    val k1 = DiagonalKernel(kernel, 3)
    val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)
    k1 + k2
  }

  val symmetrizedGaussian = symmetrizeKernel(GaussianKernel[_3D](100))

  val gpSym = GaussianProcess(zeroMean, symmetrizedGaussian)
  val sampleGpSym =  gpSym.sampleAtPoints(referenceMesh.pointSet)
  ui.show(sampleGroup, sampleGpSym, "ChangePointKernelGP_sample")

}
