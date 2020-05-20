object Tutorial_3 extends App {

  /* From meshes to deformation fields - This is a test
  Fabio Fehr
  20 May 2020
  * */

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.registration.Transformation

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)


  val ui = ScalismoUI()
  import scalismo.io.MeshIO

  val dsGroup = ui.createGroup("datasets")

  // This grabs 3 faces and displays them
  val meshFiles = new java.io.File("datasets/testFaces/").listFiles.take(3)
  val (meshes, meshViews) = meshFiles.map(meshFile => {
    val mesh = MeshIO.readMesh(meshFile).get
    val meshView = ui.show(dsGroup, mesh, "mesh")
    (mesh, meshView) // return a tuple of the mesh and the associated view
  }) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews


  //set the reference mesh
  val reference = meshes(0) // face_0 is our reference

  // Now any mesh, which is in correspondence with this reference, can be represented as a deformation field!
  // This is simply the difference between the new mesh and the reference
  val deformations : IndexedSeq[EuclideanVector[_3D]] = reference.pointSet.pointIds.map {
    id =>  meshes(1).pointSet.point(id) - reference.pointSet.point(id)
  }.toIndexedSeq

  // A deformation field in 3D is both deformation vector and the reference points
  val deformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformations)
  // NB this is discrete finite set of points = domain

  // This is the deformation field for point id =0
  deformationField(PointId(0))
  // res2: EuclideanVector[_3D] = EuclideanVector3D(
  //   -0.031402587890625,
  //   -0.24579620361328125,
  //   4.780601501464844
  // )

  // visualise the deformation field in the UI
  val deformationFieldView = ui.show(dsGroup, deformationField, "deformations")

  // removes the face3 and then makes the reference see though. It shows all the deformations
  meshViews(2).remove()
  meshViews(0).opacity = 0.3

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Deformation fields over continuous domains: ////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Now lets make the domain continuous by means of interpolation.
  val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
  val continuousDeformationField : Field[_3D, EuclideanVector[_3D]] = deformationField.interpolate(interpolator)

  // This is sext as its defined on the entire real numberline and evaluate any 3D point
  continuousDeformationField(Point(-100,-100,-100))
  // images tend to use linear or b-spline interpolators which are also good

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // The mean deformation and the mean mesh: ////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////


  val nMeshes = meshes.length

  // Maps each id to the average deformation for that id returns the deformation field
  val meanDeformations = reference.pointSet.pointIds.map( id => {

    var meanDeformationForId = EuclideanVector(0, 0, 0)

    // Loops through all meshes and gets the average
    val meanDeformations = meshes.foreach (mesh => { // loop through meshes
      val deformationAtId = mesh.pointSet.point(id) - reference.pointSet.point(id)
      meanDeformationForId += deformationAtId * (1.0 / nMeshes)
    })

    meanDeformationForId
  })

  val meanDeformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](
    reference.pointSet,
    meanDeformations.toIndexedSeq
  )

  // We can now apply the deformation to every point of the reference mesh, to obtain the mean mesh.
  val continuousMeanDeformationField = meanDeformationField.interpolate(interpolator)
  val meanTransformation = Transformation((pt : Point[_3D]) => pt + continuousMeanDeformationField(pt))

  val meanMesh = reference.transform(meanTransformation)

  ui.show(dsGroup, meanMesh, "mean mesh")
}
