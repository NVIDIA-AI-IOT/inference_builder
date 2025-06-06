    async def {{ name }}(self, request, asset_id):
        try:
            status = self._asset_manager.delete_asset(asset_id)
        except Exception as e:
            return 500, str(e)
        response = self.process_response("{{ name }}", request, {"status": status})
        return 200, response
