object Tutorial_5 extends App {

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.io.StatisticalModelIO
  import scalismo.statisticalmodel._

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Discrete and Continuous Gaussian processes ///////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // A PDM is just a discrete GP over the reference points
  val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/bfm.h5")).get
  val gp = model.gp

  val modelGroup = ui.createGroup("modelGroup")
  val ssmView = ui.show(modelGroup, model, "model")

  // we can sample as follows
  val sampleDF : DiscreteField[_3D,UnstructuredPointsDomain[_3D], EuclideanVector[_3D]]
  = model.gp.sample

  val sampleGroup = ui.createGroup("sample")
  ui.show(sampleGroup, sampleDF, "discreteSample")

  // interpolate the GP directly to make it continous (only makes sense close to the mesh)
  val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
  val contGP = model.gp.interpolate(interpolator)

  val contSample: Field[_3D, EuclideanVector[_3D]] = contGP.sample

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // From continuous to discrete: marginalization /////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //In practice, we will never work with a continuous Gaussian process directly.
  // We are always interested in the distribution on a finite set of points.
  // The real advantage of having a continuous Gaussian process is, that we can get samples at any finite
  // set of points (marginalize it) and thereby choosing the discretization according to the needs of our application.

  // All reference points
  val fullSample = contGP.sampleAtPoints(model.referenceMesh.pointSet)
  val fullSampleView = ui.show(sampleGroup, fullSample, "fullSample")

  //Only a single point
  fullSampleView.remove()
  val singlePointDomain : DiscreteDomain[_3D] =
    UnstructuredPointsDomain(IndexedSeq(model.referenceMesh.pointSet.point(PointId(8156))))
  val singlePointSample = contGP.sampleAtPoints(singlePointDomain)
  ui.show(sampleGroup, singlePointSample, "singlePointSample")

  // Only certain points (eyes)
  val referencePointSet = model.referenceMesh.pointSet
  val rightEyePt: Point[_3D] = referencePointSet.point(PointId(4281))
  val leftEyePt: Point[_3D] = referencePointSet.point(PointId(11937))
  val dom = UnstructuredPointsDomain(IndexedSeq(rightEyePt,leftEyePt))
  val marginal : DiscreteGaussianProcess[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = contGP.marginal(dom)

  // We now can sample from the eyes and get new vectors
  val sample : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = marginal.sample
  ui.show(sampleGroup, sample, "marginal_sample")

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Marginal of a statistical mesh model /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // We pass it IDs
  val noseTipModel : StatisticalMeshModel = model.marginal(IndexedSeq(PointId(8156)))

  val tipSample : TriangleMesh[_3D] = noseTipModel.sample
  // tipSample: TriangleMesh[_3D] = Triangle
  //   scalismo.common.UnstructuredPointsDomain3D@6e15e143,
  //   TriangleList(Vector())
  // )
  println("nb mesh points " + tipSample.pointSet.numberOfPoints)
  // nb mesh points 1

  // Nose marginal

  val middleNose = referencePointSet.point(PointId(8152))
  val nosePtIDs : Iterator[PointId] = referencePointSet.pointsWithId
    .filter( ptAndId => {  // yields tuples with point and ids
      val (pt, id) = ptAndId
      (pt - middleNose).norm > 40
    })
    .map(ptAndId => ptAndId._2) // extract the id's

  val noseModel = model.marginal(nosePtIDs.toIndexedSeq)
  val noseGroup = ui.createGroup("noseModel")
  ui.show(noseGroup, noseModel, "noseModel")

  // Probability of shapes and deformations:

  // How do we access how probably an instance is?
  // We use the pdf method

  val defSample = noseModel.gp.sample
  noseModel.gp.pdf(defSample)

  val defSample1 = noseModel.gp.sample
  // defSample1: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = <function1>
  val defSample2 = noseModel.gp.sample
  // defSample2: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = <function1>

  val logPDF1 = noseModel.gp.logpdf(defSample1)
  // logPDF1: Double = -11.265529462996712
  val logPDF2 = noseModel.gp.logpdf(defSample2)
  // logPDF2: Double = -17.33330521109113

  val moreOrLess = if (logPDF1 > logPDF2) "more" else "less"
  // moreOrLess: String = "more"
  println(s"defSample1 is $moreOrLess likely than defSample2")
  // defSample1 is more likely than defSample2

  
}
