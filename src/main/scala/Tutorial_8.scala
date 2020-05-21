object Tutorial_8 extends App {

  /*Posterior Shape Models
  * Fabio Fehr
  * 21 May 2020
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

  val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/bfm.h5")).get

  val modelGroup = ui.createGroup("modelGroup")
  val ssmView = ui.show(modelGroup, model, "model")

  //Fitting observed data using Gaussian process regression

  //given some observed data, we fit the model to the data and get as a result a distribution over the model parameters.
  // model = Gaussian process model of shape deformations
  // data =  observed shape deformations;
  // I.e. deformation vectors from the reference surface.

  // Now we generate a nose tip mean thats large *2
  val idNoseTip = PointId(8156)
  val noseTipReference = model.referenceMesh.pointSet.point(idNoseTip)
  val noseTipMean = model.mean.pointSet.point(idNoseTip)
  val noseTipDeformation = (noseTipReference - noseTipMean) * 2.0

  val noseTipDomain = UnstructuredPointsDomain(IndexedSeq(noseTipReference))
  val noseTipDeformationAsSeq = IndexedSeq(noseTipDeformation)
  val noseTipDeformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](noseTipDomain, noseTipDeformationAsSeq)

  val observationGroup = ui.createGroup("observation")
  ui.show(observationGroup, noseTipDeformationField, "noseTip") // deformation from target to nosetip

  // The GP assumes some uncertainty
  val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

  // the data for the regression is specified by a sequence of triples,
  // reference, the corresponding deformation vector,
  // noise at that point:
  val regressionData = IndexedSeq((noseTipReference, noseTipDeformation, noise))

  val gp : LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = model.gp.interpolate(NearestNeighborInterpolator())
  val posteriorGP : LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = LowRankGaussianProcess.regression(gp, regressionData)

  //Note that the result of the regression is again a Gaussian process, over the same domain as the original process.
  // We call this the posterior process.

  // We can now sample from this big nose posterior
  val posteriorSample : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]]
  = posteriorGP.sampleAtPoints(model.referenceMesh.pointSet)
  val posteriorSampleGroup = ui.createGroup("posteriorSamples")
  for (i <- 0 until 2) {
    ui.show(posteriorSampleGroup, posteriorSample, "posteriorSample")
  }

  // Posterior of a StatisticalMeshModel:

  val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.01)
  val pointOnLargeNose = noseTipReference + noseTipDeformation
  val discreteTrainingData = IndexedSeq((PointId(8156), pointOnLargeNose, littleNoise))
  val meshModelPosterior : StatisticalMeshModel = model.posterior(discreteTrainingData)

  // Notice all these faces have groot snozzes
  val posteriorModelGroup = ui.createGroup("posteriorModel")
  ui.show(posteriorModelGroup, meshModelPosterior, "NoseyModel")

//  Landmark uncertainty:

  // if we have large uncertainty (5 times bugger ) then we see loads of change in the nose size.
  //
  val largeNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 5.0)
  val discreteTrainingDataLargeNoise = IndexedSeq((PointId(8156), pointOnLargeNose, largeNoise))
  val discretePosteriorLargeNoise = model.posterior(discreteTrainingDataLargeNoise)
  val posteriorGroupLargeNoise = ui.createGroup("posteriorLargeNoise")
  ui.show(posteriorGroupLargeNoise, discretePosteriorLargeNoise, "NoisyNoseyModel")

  println("fin")

}
