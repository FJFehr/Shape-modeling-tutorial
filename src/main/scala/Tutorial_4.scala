object Tutorial_4 extends App {

  /* GPs and PDMs
  Fabio Fehr
  20.05.2020
  * */

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.io.StatismoIO
  import scalismo.statisticalmodel._

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  // This is a probability distribution of face meshes.
  val faceModel = StatismoIO.readStatismoMeshModel(new java.io.File("datasets/bfm.h5")).get
  val modelGroup = ui.createGroup("model")

  // lets visualise the mean shape
  val sampleGroup = ui.createGroup("samples")
  val meanFace : TriangleMesh[_3D] = faceModel.mean
  ui.show(sampleGroup, meanFace, "meanFace")

  // lets sample a face
  val sampledFace : TriangleMesh[_3D] = faceModel.sample
  ui.show(sampleGroup, sampledFace, "randomFace")

  // Do not be fooled!
  // a PDM is represented as a triangle mesh
  // On this mesh a Gaussian Process over deformation fields is defined

  val reference : TriangleMesh[_3D] = faceModel.referenceMesh
  val faceGP : DiscreteLowRankGaussianProcess[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = faceModel.gp


  /* The GP type signature can tell us the following:
    It is a DiscreteGaussianProcess. This means, the function, which the process models are defined on a discrete, finite set of points.
    It is defined in 3D Space (indicated by the type parameter _3D)
    Its domain of the modeled functions is a UnstructuredPointsDomain (namely the points of the reference mesh)
    The values of the modeled functions are vectors (more precisely, they are of type EuclideanVector).
    It is represented using a low-rank approximation.
    */

  // We are just sampling from a GP to get a new deformation vector
  val meanDeformation : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = faceGP.mean
  val sampleDeformation : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = faceGP.sample

  // lets visualise the mean deformation. This goes from
  ui.show(sampleGroup, meanDeformation, "meanField")

  ui.show(modelGroup, reference, "referenceFace")

  // 1) the mean deformation field is obtained (by calling faceModel.gp.mean)
  // 2) the mean deformation field is then used to deform the reference mesh (faceModel.referenceMesh)
  // into the triangle Mesh representing the mean face
}
