#if !defined(_MESH_ID_H_)
#define _MESH_ID_H_


namespace implementation
{


template<typename IDType, typename GlobalIndexType>
class IDProvider
{
public:
	IDProvider()                   : m_id(-1) {}

	const IDType &getID() const    { assert(m_id >= 0); return m_id; }

protected:
	void setID(GlobalIndexType id) { m_id = id; }

private:
	IDType m_id;
};

template<typename GlobalIndexType>
class IDProvider<void, GlobalIndexType>
{
protected:
	void setID(GlobalIndexType) {}
};


} // namespace implementation


#endif
