object Tutorial_13 extends App {

  /*ASM in Scalismo!
  * we can use it to learn active shape models from a set of images and corresponding contour,
  * we can save these models, and we can use them to fit images. We will focus on the fitting
  * */

  import scalismo.geometry._
  import scalismo.ui.api._
  import scalismo.registration._
  import scalismo.mesh.TriangleMesh
  import scalismo.statisticalmodel.asm._
  import scalismo.io.{ActiveShapeModelIO, ImageIO}
  import breeze.linalg.{DenseVector}


  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)


  val ui = ScalismoUI()

  // This is both a shape model and intensity model!
  val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("datasets/femur-asm.h5")).get

  //An ActiveShapeModel instance in Scalismo is a combination of a statistical shape model and an intensity model.
  // Using the method statisticalModel, we can obtain the shape model part. Let’s visualize this model:

  val modelGroup = ui.createGroup("modelGroup")
  val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

  // Now for the intensity part. These are profiles on certain verticies. Profiles have Probability distributions

  val profiles = asm.profiles
  profiles.map(profile => {
    val pointId = profile.pointId
    val distribution = profile.distribution
  })

  // We use this to find the points in an image

  val image = ImageIO.read3DScalarImage[Short](new java.io.File("datasets/femur-image.nii")).get.map(_.toFloat)
  val targetGroup = ui.createGroup("target")

  val imageView = ui.show(targetGroup, image, "image")

  // This smooths/gradient transforms etc to make it work for asm
  val preprocessedImage = asm.preprocessor(image)

  // Now extract the features
  val point1 = image.domain.origin + EuclideanVector(10.0, 10.0, 10.0)
  val profile = asm.profiles.head
  val feature1 : DenseVector[Double] = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get

  // Here we specified the preprocessed image,
  // a point in the image where whe want the evaluate the feature vector,
  // a mesh instance
  // and a point id for the mesh.

  // The mesh and ID are needed to focus where to extract features. We then get the likelihood of the corresponding point
  // corresponds to a to a given profile point

  val point2 = image.domain.origin + EuclideanVector(20.0, 10.0, 10.0)
  val featureVec1 = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get
  val featureVec2 = asm.featureExtractor(preprocessedImage, point2, asm.statisticalModel.mean, profile.pointId).get

  val probabilityPoint1 = profile.distribution.logpdf(featureVec1)
  val probabilityPoint2 = profile.distribution.logpdf(featureVec2)
  // Now we can decide which point is more likely! This is the idea of ASM


  //  The original Active Shape Model Fitting  by Cootes and Taylor.

  // To configure the fitting process, we need to set up a search method,
  // which searches for a given model point, corresponding points in the image. Once we have these candidate correspondences
  // We can just do ICP

  // One search strategy that is already implemented in Scalismo is to search along the normal direction of a model point.
  val searchSampler = NormalDirectionSearchPointSampler(numberOfPoints = 100, searchDistance = 3)

  val config = FittingConfiguration(featureDistanceThreshold = 3, pointDistanceThreshold = 5, modelCoefficientBounds = 3)
  // The first parameter determines how far away (as measured by the mahalanobis distance)
  // this case points which are more than 5 standard deviations aways are not chosen as corresponding points.
  //how large coefficients of the model can become. Whenever a model parameter is larger than this threshold,
  // it will be set back to this maximal value. This introduces a regularization

  // ASM fitting algorithm optimizes both the pose (as defined by a rigid transformation) and the shape.

  // make sure we rotate around a reasonable center point
  val modelBoundingBox = asm.statisticalModel.referenceMesh.boundingBox
  val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

  //To initialize the fitting process, we also need to set up the initial transformation: ALL BLANKS
  val translationTransformation = TranslationTransform(EuclideanVector(0, 0, 0))
  val rotationTransformation = RotationTransform(0, 0, 0, rotationCenter)
  val initialRigidTransformation = RigidTransformation(translationTransformation, rotationTransformation)
  val initialModelCoefficients = DenseVector.zeros[Double](asm.statisticalModel.rank)
  val initialTransformation = ModelTransformations(initialModelCoefficients, initialRigidTransformation)

  // More parameters
  val numberOfIterations = 20
  val asmIterator = asm.fitIterator(image, searchSampler, numberOfIterations, config, initialTransformation)

  // visualise each update
  val asmIteratorWithVisualization = asmIterator.map(it => {
    it match {
      case scala.util.Success(iterationResult) => {
        modelView.shapeModelTransformationView.poseTransformationView.transformation = iterationResult.transformations.rigidTransform
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = iterationResult.transformations.coefficients
      }
      case scala.util.Failure(error) => System.out.println(error.getMessage)
    }
    it
  })

  // To run the fitting, and get the result, we finally consume the iterator:
  val result = asmIteratorWithVisualization.toIndexedSeq.last
  val finalMesh = result.get.mesh

  // Evaluating the likelihood of a model instance under the image

  // In the previous section we have used the intensity distribution to find the best corresponding
  // image point to a given point in the model. how well a model fits an image?
  // extend the method used above to compute the likelihood for all profile points of an Active Shape Model.

  //  Assuming independence, the joint probability is just the product of the probability
  //  at the individual profile points. In order not to get too extreme values, we use log
  //  probabilities here (and consequently the product becomes a sum). MATHHHHH

  def likelihoodForMesh(asm : ActiveShapeModel, mesh : TriangleMesh[_3D], preprocessedImage: PreprocessedImage) : Double = {

    val ids = asm.profiles.ids

    val likelihoods = for (id <- ids) yield {
      val profile = asm.profiles(id)
      val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
      val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).get
      profile.distribution.logpdf(featureAtPoint)
    }
    likelihoods.sum
  }

  // We now can calculate how likeliy it is to correspond to a given image
  val sampleMesh1 = asm.statisticalModel.sample
  val sampleMesh2 = asm.statisticalModel.sample
  println("Likelihood for mesh 1 = " + likelihoodForMesh(asm, sampleMesh1, preprocessedImage))
  println("Likelihood for mesh 2 = " + likelihoodForMesh(asm, sampleMesh2, preprocessedImage))

  // Thus mesh 1 (is larger) and thus more likely
}
