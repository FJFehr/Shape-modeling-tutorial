object Tutorial_12 extends App {

  /*Parametric, non-rigid registration
  *
  * formulating the registration problem as an optimization problem, which we optimize using gradient-based optimization
  * Image to image and surface to surface registration is possible now
  * */

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.registration._
  import scalismo.io.{MeshIO}
  import scalismo.numerics._
  import scalismo.kernels._
  import scalismo.statisticalmodel._
  import breeze.linalg.{DenseVector}

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  // Load the mesh
  val referenceMesh = MeshIO.readMesh(new java.io.File("datasets/quickstart/facemesh.ply")).get

  val modelGroup = ui.createGroup("model")
  val refMeshView = ui.show(modelGroup, referenceMesh, "referenceMesh")
  refMeshView.color = java.awt.Color.RED

  // Build the GP

  val mean = VectorField(RealSpace[_3D], (_ : Point[_3D]) => EuclideanVector.zeros[_3D])
  val kernel = DiagonalKernel[_3D](GaussianKernel(sigma = 70) * 50.0, outputDim = 3)
  val gp = GaussianProcess(mean, kernel)

  // get a low rank approx

  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    referenceMesh.pointSet,
    gp,
    relativeTolerance = 0.05,
    interpolator = NearestNeighborInterpolator()
  )

  // Visualise the effect?
  val gpView = ui.addTransformation(modelGroup, lowRankGP, "gp")

  // Registration

  val targetGroup = ui.createGroup("target")
  val targetMesh = MeshIO.readMesh(new java.io.File("datasets/quickstart/face-2.ply")).get
  val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")

  // To define registration we need four things

  // 1) a transformation space that models the possible transformations of the reference surface (or the ambient space)
  // 2) a metric to measure the distance between the model (the deformed reference mesh) an the target surface.
  // 3) a regularizer, which penalizes unlikely transformations.
  // 4) an optimizer.

  // 1. The transformation space is defined by the GP
  val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)

  // 2. Metric we shall use simple mean squared
  // The sampler determines the points where the metric is evaluated.
  // In our case we choose uniformely sampled points on the reference mesh.

  // image metrics thus we transform the surface to a distance image
  val fixedImage = referenceMesh.operations.toDistanceImage
  val movingImage = targetMesh.operations.toDistanceImage
  val sampler = FixedPointsUniformMeshSampler3D(referenceMesh, numberOfPoints = 1000)
  val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)

  // 3. Regularizer with an L2 norm
  val regularizer = L2Regularizer(transformationSpace)

  // 4. Optimisation Broyden–Fletcher–Goldfarb–Shanno. Due to its resulting linear memory requirement,
  // the L-BFGS method is particularly well suited for optimization problems with many variables
  val optimizer = LBFGSOptimizer(maxNumberOfIterations = 100)

  // Registration object with 2,3,4
  val registration = Registration(metric, regularizer, regularizationWeight = 1e-5, optimizer)

  //Since its iterative we define an iterator
  val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank) // zerovector
  val registrationIterator = registration.iterator(initialCoefficients)

  // Before running the registration, we change the iterator such that it prints in each iteration to
  // current objective value, and updates the visualization.
  // This lets us visually inspect the progress of the registration procedure.

  val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
    println(s"object value in iteration $itnum is ${it.value}")
    gpView.coefficients = it.parameters
    it
  }
  // This code above simply augments the old iterator by adding a visualisation
  // The actual registration is executed once we “consume” the iterator.
  // This can, for example be achieved by converting it to a sequence.
  // The resulting sequence holds all the intermediate states of the registration.
  // We are usually only interested in the last one:

  val registrationResult = visualizingRegistrationIterator.toSeq.last

  // We should now see in the GUI, the ref morph into our target
  val registrationTransformation = transformationSpace.transformForParameters(registrationResult.parameters)
  val fittedMesh = referenceMesh.transform(registrationTransformation)

  // The fittedMesh that we obtained above is a surface that approximates the target surface.
  // It is the best representation of the target from the model. this approx is often sufficient

  // If its not sufficient we can extract an exact representation of the target mesh using a projection
  val targetMeshOperations = targetMesh.operations
  val projection = (pt : Point[_3D]) => {
    targetMeshOperations.closestPointOnSurface(pt).point
  }

  // Now we can get a mapping for each point
  val finalTransformation = registrationTransformation.andThen(projection)

  val projectedMesh = referenceMesh.transform(finalTransformation)
  val resultGroup = ui.createGroup("result")
  val projectionView = ui.show(resultGroup, projectedMesh, "projection")


  // What about complex shapes?

  //This registration procedure outlined above works reasonably well for simple cases.
  // In complex cases, in particular if you have large shape variations, you may find it
  //  difficult to find a suitable regularization weight.
  // Weight is large then smooth mesh not fitting closely
  // Weight is small then bad correspondences.
  // Thus we repear this process with different weights

  case class RegistrationParameters(regularizationWeight : Double, numberOfIterations : Int, numberOfSampledPoints : Int)

  def doRegistration(
                      lowRankGP : LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
                      referenceMesh : TriangleMesh[_3D],
                      targetmesh : TriangleMesh[_3D],
                      registrationParameters : RegistrationParameters,
                      initialCoefficients : DenseVector[Double]
                    ) : DenseVector[Double] =
  {
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = FixedPointsUniformMeshSampler3D(
      referenceMesh,
      registrationParameters.numberOfSampledPoints
    )
    val metric = MeanSquaresMetric(
      fixedImage,
      movingImage,
      transformationSpace,
      sampler
    )
    val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
    val regularizer = L2Regularizer(transformationSpace)
    val registration = Registration(
      metric,
      regularizer,
      registrationParameters.regularizationWeight,
      optimizer
    )
    val registrationIterator = registration.iterator(initialCoefficients)
    val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
      println(s"object value in iteration $itnum is ${it.value}")
      gpView.coefficients = it.parameters
      it
    }
    val registrationResult = visualizingRegistrationIterator.toSeq.last
    registrationResult.parameters
  }

  // Now we define the parameters and run the registration

  val registrationParameters = Seq(
    RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 20, numberOfSampledPoints = 1000),
    RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 30, numberOfSampledPoints = 1000),
    RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 40, numberOfSampledPoints = 2000),
    RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50, numberOfSampledPoints = 4000)
  )

  // So it takes registration parameters and uses fold left for each of them
  // initial coefficients (zeros) then defines inputs as modelCoefficients and regParameters and applys them to doReg
  val finalCoefficients = registrationParameters.foldLeft(initialCoefficients)((modelCoefficients, regParameters) =>
    doRegistration(lowRankGP, referenceMesh, targetMesh, regParameters, modelCoefficients))

  println("fin")
}

