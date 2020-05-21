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

  // few data thus small model
  val smallModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/model.h5")).get

  // Enlarging the flexibility of a shape model

  //o increase the shape variability of the model,
  // we add smooth some additional smooth shape deformations, modelled by a GP with symmetric Gaussian kernel.

  val scalarValuedKernel = GaussianKernel[_3D](70) * 10.0

  case class XmirroredKernel(kernel : PDKernel[_3D]) extends PDKernel[_3D] {
    override def domain = RealSpace[_3D]
    override def k(x: Point[_3D], y: Point[_3D]) = kernel(Point(x(0) * -1f ,x(1), x(2)), y)
  }

  def symmetrizeKernel(kernel : PDKernel[_3D]) : MatrixValuedPDKernel[_3D] = {
    val xmirrored = XmirroredKernel(kernel)
    val k1 = DiagonalKernel(kernel, 3)
    val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)
    k1 + k2
  }

  val gp = GaussianProcess[_3D, EuclideanVector[_3D]](symmetrizeKernel(scalarValuedKernel))
  val lowrankGP = LowRankGaussianProcess.approximateGPCholesky(
    smallModel.referenceMesh.pointSet,
    gp,
    relativeTolerance = 0.01,
    interpolator = NearestNeighborInterpolator())

  val model = StatisticalMeshModel.augmentModel(smallModel, lowrankGP)

  val modelGroup = ui.createGroup("face model")
  val ssmView = ui.show(modelGroup, model, "model")
}
