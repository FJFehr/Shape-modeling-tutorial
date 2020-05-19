/*
* Hello Scalismo
* This is a basic tutorial through Scala and using the IDE Intellij
*
* Things to remember:
* * extends App so that its excecutable
* * add the following lines to the build.sbt

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")
libraryDependencies ++= Seq("ch.unibas.cs.gravis" %% "scalismo-ui" % "0.14-RC1")

* MUST USE THE CORRECT JDK adoptopenJDK
* */

object Tutorial_1 extends App {

  // loads all the dependencies to native C++ libraries
  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  // visualize the objects we create.
  import scalismo.ui.api.ScalismoUI
  val ui = ScalismoUI()

  // Imports
  import scalismo.mesh.TriangleMesh // the mesh class
  import scalismo.io.MeshIO // to read meshes
  import scalismo.common.PointId // to refer to points by id
  import scalismo.mesh.TriangleId // to refer to triangles by id
  import scalismo.geometry._3D // indicates that we work in 3D space

  // Actually fetching the face
  val mesh : TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("../datasets/Paola.ply")).get

  val paolaGroup = ui.createGroup("paola")
  val meshView = ui.show(paolaGroup, mesh, "Paola")

  println("first point " + mesh.pointSet.point(PointId(0)))
  // first point Point3D(162.2697296142578,-11.115056991577148,301.18719482421875)

  println("first cell " + mesh.triangulation.triangle(TriangleId(0)))
  // first cell TriangleCell(PointId(0),PointId(1),PointId(2))

  val pointCloudView = ui.show(paolaGroup, mesh.pointSet, "pointCloud")

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Working with vectors ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  import scalismo.geometry.{Point}
  import scalismo.geometry.{EuclideanVector}

  // lets define two points
  val p1 : Point[_3D] = Point(4.0, 5.0, 6.0)
  val p2 : Point[_3D] = Point(1.0, 2.0, 3.0)

  // Here shows the difference of 2 points returns a vector
  val v1 : EuclideanVector[_3D] = Point(4.0, 5.0, 6.0) - Point(1.0 , 2.0, 3.0)

  // The sum of a point with a vector yields a new point:
  val p3 : Point[_3D] = p1 + v1

  // Which can be converted to a vector
  val v2 : EuclideanVector[_3D] = p1.toVector

  // and back from a vector to a point
  val v3 : Point[_3D] = v1.toPoint

  // Now lets use these ideas to find the center of mass
   val pointList = Seq(Point(4.0, 5.0, 6.0), Point(1.0, 2.0, 3.0), Point(14.0, 15.0, 16.0), Point(7.0, 8.0, 9.0), Point(
    10.0, 11.0, 12.0))

  val vectors = pointList.map{p : Point[_3D] => p.toVector}  // use map to turn points into

  val vectorSum = vectors.reduce{ (v1, v2) => v1 + v2} // sum up all vectors in the collection
  val centerV: EuclideanVector[_3D] = vectorSum * (1.0 / pointList.length ) // divide the sum by the number of points

  println("Center point =  " + centerV)
  //EuclideanVector3D(7.2,8.200000000000001,9.200000000000001)

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Working with meshes /////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  import scalismo.io.StatisticalModelIO // to read statistical shape models
  import scalismo.statisticalmodel.StatisticalMeshModel // the statistical shape models

  val faceModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("../datasets/bfm.h5")).get
  val faceModelView = ui.show(faceModel, "faceModel")

  val randomFace1 : TriangleMesh[_3D] = faceModel.sample
  val randomFace2 : TriangleMesh[_3D] = faceModel.sample
  val randomFace3 : TriangleMesh[_3D] = faceModel.sample

  val meshView1 = ui.show(randomFace1, "rando1")
  val meshView2 = ui.show(randomFace2, "rando2")
  val meshView3 = ui.show(randomFace3, "rando3")

  import scalismo.ui.api.LandmarkView
  import scalismo.geometry.Landmark

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Working with landmarks //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  // to click a landmark we need to use a filter
  val matchingLandmarkViews : Seq[LandmarkView] = ui.filter[LandmarkView](paolaGroup, (l : LandmarkView) => l.name == "noseLM")
  val matchingLandmarks : Seq[Landmark[_3D]] = matchingLandmarkViews.map(lmView => lmView.landmark)

  val landmarkId : String = matchingLandmarks.head.id
  val landmarkPosition : Point[_3D] = matchingLandmarks.head.point
}
