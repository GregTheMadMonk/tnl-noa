#if !defined(_MESH_POINTER_H_)
#define _MESH_POINTER_H_


namespace implementation
{


template<typename> class Mesh;

template<typename MeshConfigTag>
class MeshPointerProvider
{
	typedef Mesh<MeshConfigTag> MeshType;

public:
	MeshPointerProvider()        : m_mesh(0) {}

	MeshType &getMesh() const    { assert(m_mesh); return *m_mesh; }

protected:
	void setMesh(MeshType &mesh) { m_mesh = &mesh; }

private:
	MeshType *m_mesh;
};


} // namespace implementation


#endif
