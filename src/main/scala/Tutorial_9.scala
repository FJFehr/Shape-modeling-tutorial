object Tutorial_9 extends App {

  // Shape completion using Gaussian process regression

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.io.{StatisticalModelIO, MeshIO, LandmarkIO}
  import scalismo.statisticalmodel._
  import scalismo.numerics.UniformMeshSampler3D
  import scalismo.kernels._
  import breeze.linalg.{DenseMatrix, DenseVector}

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  val noseless = MeshIO.readMesh(new java.io.File("datasets/noseless.ply")).get

  // Try and fill in the nose!
  val targetGroup = ui.createGroup("target")
  ui.show(targetGroup, noseless,"noseless")

  // few data (10 shapes) thus small model
  val smallModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/model.h5")).get

  // Enlarging the flexibility of a shape model

  // To increase the shape variability of the model,
  // we add smooth some additional smooth shape deformations, modelled by a GP with symmetric Gaussian kernel.

  // Simple Gaussian Kernel
  val scalarValuedKernel = GaussianKernel[_3D](70) * 10.0

  case class XmirroredKernel(kernel : PDKernel[_3D]) extends PDKernel[_3D] {
    override def domain = RealSpace[_3D]
    override def k(x: Point[_3D], y: Point[_3D]) = kernel(Point(x(0) * -1f ,x(1), x(2)), y)
  }

  // force it to be symmetric
  def symmetrizeKernel(kernel : PDKernel[_3D]) : MatrixValuedPDKernel[_3D] = {
    val xmirrored = XmirroredKernel(kernel)
    val k1 = DiagonalKernel(kernel, 3)
    val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)
    k1 + k2
  }

  // Makes a GP with domain = reference, uses an error metric of 0.01 instead of a set amount of eigen components
  val gp = GaussianProcess[_3D, EuclideanVector[_3D]](symmetrizeKernel(scalarValuedKernel))
  val lowrankGP = LowRankGaussianProcess.approximateGPCholesky(
    smallModel.referenceMesh.pointSet,
    gp,
    relativeTolerance = 0.01,
    interpolator = NearestNeighborInterpolator())

  val model = StatisticalMeshModel.augmentModel(smallModel, lowrankGP)

  val modelGroup = ui.createGroup("face model")
  val ssmView = ui.show(modelGroup, model, "model")

  // for reconstrunction its advised to make the models more flexible as this only had 10 faces (small model)
  // It gives the model some extra slack to account for bias in the data and explain minor shape variations,
  // which have not been prominent in the dataset.

  // Reconstruction
  //1) We fit the face model to the given partial face using Gaussian process regression.
  //2) We restrict the model to the nose part by marginalizing and select a suitable nose shape.
  //3) We choose a suitable nose from the model

  // Get reference landmarks
  val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/modelLandmarks.json")).get
  val referencePoints : Seq[Point[_3D]] = referenceLandmarks.map(lm => lm.point)
  val referenceLandmarkViews = referenceLandmarks.map(lm => ui.show(modelGroup, lm, s"lm-${lm.id}"))


  val noselessLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/noselessLandmarks.json")).get
  val noselessPoints : Seq[Point[_3D]] = noselessLandmarks.map(lm => lm.point)
  val noselessLandmarkViews = noselessLandmarks.map(lm => ui.show(targetGroup, lm, s"lm-${lm.id}"))


  val domain = UnstructuredPointsDomain(referencePoints.toIndexedSeq)
  val deformations = (0 until referencePoints.size).map(i => noselessPoints(i) - referencePoints(i) )
  val defField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](domain, deformations)
  ui.show(modelGroup, defField, "partial_Field")

  // Perform a GP regression
  val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.5)

  val regressionData = for ((refPoint, noselessPoint) <- referencePoints zip noselessPoints) yield {
    val refPointId = model.referenceMesh.pointSet.findClosestPoint(refPoint).id
    (refPointId, noselessPoint, littleNoise)
  }

  val posterior = model.posterior(regressionData.toIndexedSeq)

  val posteriorGroup = ui.createGroup("posterior-model")
  ui.show(posteriorGroup, posterior, "posterior")

  // Now we marginalise over the nose only
  val nosePtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
    (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(PointId(8152))).norm <= 42
  }

  val posteriorNoseModel = posterior.marginal(nosePtIDs.toIndexedSeq)

  val posteriorNoseGroup = ui.createGroup("posterior-nose-model")
  ui.show(posteriorNoseGroup, posteriorNoseModel, "posteriorNoseModel")

  // The mean will be the best reconstruntion
  val bestReconstruction = posteriorNoseModel.mean

  println("fin")
}
