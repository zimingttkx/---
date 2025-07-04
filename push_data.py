from pymongo import MongoClient

# 1. 将 URI 修改为本地数据库的地址
#    The URI is changed to the address of the local database.
uri = "mongodb://localhost:27017/"

# 2. 创建客户端实例
#    Create a client instance. For local connections, the server_api parameter is usually not needed.
client = MongoClient(uri)

# 3. 发送一个 "ping" 来确认连接成功
#    Send a "ping" to confirm a successful connection.
try:
    client.admin.command('ping')
    print("您已成功连接到本地 MongoDB！")
    print("You successfully connected to your local MongoDB!")
except Exception as e:
    print(e)

# 4. (可选) 查看您本地的数据库列表
#    (Optional) List the databases on your local instance.
print("\n您本地的数据库列表 (List of your local databases):")
print(client.list_database_names())

# 关闭连接
# Close the connection
client.close()