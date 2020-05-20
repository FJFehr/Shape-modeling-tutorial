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

  val meshFiles = new java.io.File("datasets/testFaces/").listFiles.take(3)
  val (meshes, meshViews) = meshFiles.map(meshFile => {
    val mesh = MeshIO.readMesh(meshFile).get
    val meshView = ui.show(dsGroup, mesh, "mesh")
    (mesh, meshView) // return a tuple of the mesh and the associated view
  }) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews


}
